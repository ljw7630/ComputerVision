#ifdef _CH_
#pragma package <opencv>
#endif

#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include "../utilities.h"
#include <vector>
#include <sstream>

#define NUM_IMAGES 5
#define NUM_TEMPLATE_IMAGES 10

using namespace std;

IplImage * cut_image(IplImage *source, CvRect rect);
void find_bounding_rectangle(CvSeq * redComponent, CvRect &rect);
void recognition_handler(const IplImage *source, CvSeq * seq, const vector<IplImage *> templates, IplImage * target);

// Locate the red pixels in the source image.
void find_red_points( IplImage* source, IplImage* result, IplImage* temp )
{
	// TO DO:  Write code to select all the red road sign points.  You may need to clean up the result
	//        using mathematical morphology.  The result should be a binary image with the selected red
	//        points as white points.  The temp image passed may be used in your processing.

	unsigned char white_pixel[4] = {255,255,255,0};
	unsigned char black_pixel[4] = {0,0,0,0};
	int width_step=source->widthStep;
	int pixel_step=source->widthStep/source->width;
	IplImage * source_temp = cvCloneImage(source);

	/* smooth the image to reduce noise */
	cvSmooth(source, source_temp, CV_GAUSSIAN, 9, 9);

	/* convert the color space from BGR to HSV */
	cvCvtColor(source, temp, CV_BGR2HSV);

	/* carefully select hue value, saturation and value */
	const int hue_range_low = 8;
	const int hue_range_high = 155;
	const int saturation_range = 54;
	const int value_range = 55;

	/* for each row */
	for(int row = 0;row<temp->height;++row)
	{
		/* for each column */
		for(int col = 0; col< temp->width; ++ col)
		{
			/* get the current pixel */
			unsigned char* curr_point = GETPIXELPTRMACRO( temp, col, row, width_step, pixel_step );

			/* if the current pixel not match all our threshold
			 * we set it black (not black) */
			if (
				(curr_point[0] > hue_range_low && curr_point[0] < hue_range_high)
				|| (curr_point[1] < saturation_range)
				|| (curr_point[2] < value_range)
				)
			{
				PUTPIXELMACRO( result, col, row, black_pixel, width_step, pixel_step, temp->nChannels );
			}
			else /* otherwise, we think it as red */
			{
				PUTPIXELMACRO( result, col, row, white_pixel, width_step, pixel_step, temp->nChannels );
			}
		}
	}
	
	/* use open and close to make the shape smooth */
	cvMorphologyEx(result, result, NULL,NULL,CV_MOP_OPEN,1);
	cvMorphologyEx(result, result, NULL,NULL,CV_MOP_CLOSE,1);
}

CvSeq* connected_components( IplImage* source, IplImage* result )
{
	IplImage* binary_image = cvCreateImage( cvGetSize(source), 8, 1 );
	cvConvertImage( source, binary_image );
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contours = 0;
	cvThreshold( binary_image, binary_image, 1, 255, CV_THRESH_BINARY );
	cvFindContours( binary_image, storage, &contours, sizeof(CvContour),	CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	if (result)
	{
		cvZero( result );
		for(CvSeq* contour = contours ; contour != 0; contour = contour->h_next )
		{
			// small connected components will be ignored
			if(contour->total < 10)
			{
				cvDrawContours( source, contour, cvScalarAll(0), cvScalarAll(0), -1, CV_FILLED, 8);
				continue;
			}
			CvScalar color = CV_RGB( rand()&255, rand()&255, rand()&255 );
			/* replace CV_FILLED with 1 to see the outlines */
			cvDrawContours( result, contour, color, color, -1, CV_FILLED, 8 );
		}
	}
	return contours;
}

void invert_image( IplImage* source, IplImage* result )
{
	// TO DO:  Write code to invert all points in the source image (i.e. for each channel for each pixel in the result
	//        image the value should be 255 less the corresponding value in the source image).

	/* I use opencv function to invert the image 
	 * alternatively, I can use two for-loops 
	 */

	cvNot(source, result);
}

// Assumes a 1D histogram of 256 elements.
int determine_optimal_threshold( CvHistogram* hist )
{
	// TO DO:  Given a 1-D CvHistogram you need to determine and return the optimal threshold value.

	// NOTES:  Assume there are 256 elements in the histogram.
	//         To get the histogram value at index i
	//            int histogram_value_at_i = ((int) *cvGetHistValue_1D(hist, i));

	/* here I use binary search to look for optimal threshold 
	 * algorithm: Iterative (Optimal) threshold selection
	 */

	/* the starting point is the middle of the histogram, 126 */
	int curr_round_value = 126;

	/* used to store the binary search result */
	int last_round_value = -1;

	int t = 0;

	int background_pixels = 0;
	int object_pixels = 0;
	int sum_background = 0;
	int sum_foreground = 0;

	/* at first, we assume the optimal threshold is in the middle of the histogram 
	 * then calculate the foreground value and background value */

	while(true)
	{
		for(int i=0;i<256;++i)
		{
			int histogram_value_at_i = (int) *cvGetHistValue_1D(hist, i);
			if(curr_round_value > histogram_value_at_i)
			{
				object_pixels+=histogram_value_at_i;
				sum_foreground += histogram_value_at_i * i;
			}
			else
			{
				background_pixels+=histogram_value_at_i;
				sum_background += histogram_value_at_i * i;
			}
		}
		curr_round_value = 
		( 
			(float)sum_background/background_pixels
			+ (float)sum_foreground/object_pixels
		) / 2;

		/* break when the value is stable (foreground and background is balanced) */
		if(last_round_value == curr_round_value)
			break;
		else
			last_round_value = curr_round_value;
	}
	if(curr_round_value<0)
	{
		getchar();
	}
	return curr_round_value;
}

void apply_threshold_with_mask(IplImage* grayscale_image,IplImage* result_image,IplImage* mask_image,int threshold)
{
	// TO DO:  Apply binary thresholding to those points in the passed grayscale_image which correspond to non-zero
	//        points in the passed mask_image.  The binary results (0 or 255) should be stored in the result_image.

	/* set white_pixel and black_pixel in BGR color space */
	unsigned char white_pixel[4] = {255,255,255,0};
	unsigned char black_pixel[4] = {0,0,0,0};

	/* get the widthStep and pixel_step */
	int width_step_gray = mask_image->widthStep;
	int pixel_step_gray = mask_image->widthStep/mask_image->width;
	int width_step_rgb =result_image->widthStep;
	int pixel_step_rgb =result_image->widthStep/result_image->width;

	// IplImage * temp = cvCloneImage(result_image);

	/* for each row*/
	for(int row = 0;row<mask_image->height;++row)
	{
		/* for each column */
		for(int col = 0; col< mask_image->width; ++ col)
		{
			/* get the point in (col, row) from mask_image */
			unsigned char* correct_point = GETPIXELPTRMACRO( mask_image, col, row, width_step_gray, pixel_step_gray );

			/* if it's white pixel (means it's red in original image) */
			if( (*correct_point) == 255)
			{
				unsigned char* curr_point = GETPIXELPTRMACRO( grayscale_image, col, row, width_step_gray, pixel_step_gray );

				/* if the current point is greater an optimal threshold, put it as white (foreground) */
				if((*curr_point) > threshold )
				{
					PUTPIXELMACRO( result_image, col, row, white_pixel, width_step_rgb, pixel_step_rgb, result_image->nChannels );
				}
				else /* otherwise, put it as black (background) */
				{
					PUTPIXELMACRO( result_image, col, row, black_pixel, width_step_rgb, pixel_step_rgb, result_image->nChannels );
				}
			}
		}
	}
}

void determine_optimal_sign_classification( IplImage* original_image, IplImage* red_point_image, CvSeq* red_components, CvSeq* background_components, IplImage* result_image )
{
	int width_step=original_image->widthStep;
	int pixel_step=original_image->widthStep/original_image->width;
	IplImage* mask_image = cvCreateImage( cvGetSize(original_image), 8, 1 );
	IplImage* grayscale_image = cvCreateImage( cvGetSize(original_image), 8, 1 );
	cvConvertImage( original_image, grayscale_image );
	IplImage* thresholded_image = cvCreateImage( cvGetSize(original_image), 8, 1 );
	cvZero( thresholded_image );
	cvZero( result_image );
	int row=0,col=0;
	CvSeq* curr_red_region = red_components;
	// For every connected red component
	while (curr_red_region != NULL)
	{
		cvZero( mask_image );
		CvScalar color = CV_RGB( 255, 255, 255 );
		CvScalar mask_value = cvScalar( 255 );
		// Determine which background components are contained within the red component (i.e. holes)
		//  and create a binary mask of those background components.
		CvSeq* curr_background_region = curr_red_region->v_next;
		if (curr_background_region != NULL)
		{
			while (curr_background_region != NULL)
			{
				cvDrawContours( mask_image, curr_background_region, mask_value, mask_value, -1, CV_FILLED, 8 );
				cvDrawContours( result_image, curr_background_region, color, color, -1, CV_FILLED, 8 );
				curr_background_region = curr_background_region->h_next;
			}
			int hist_size=256;
			CvHistogram* hist = cvCreateHist( 1, &hist_size, CV_HIST_ARRAY );
			cvCalcHist( &grayscale_image, hist, 0, mask_image );
			// Determine an optimal threshold on the points within those components (using the mask)
			int optimal_threshold = determine_optimal_threshold( hist );

			//cout << "threshold: " << optimal_threshold << endl;

			apply_threshold_with_mask(grayscale_image,result_image,mask_image,optimal_threshold);
		}
		curr_red_region = curr_red_region->h_next;
	}

	for (row=0; row < result_image->height; row++)
	{
		unsigned char* curr_red = GETPIXELPTRMACRO( red_point_image, 0, row, width_step, pixel_step );
		unsigned char* curr_result = GETPIXELPTRMACRO( result_image, 0, row, width_step, pixel_step );
		for (col=0; col < result_image->width; col++)
		{
			curr_red += pixel_step;
			curr_result += pixel_step;
			if (curr_red[0] > 0)
				curr_result[2] = 255;
		}
	}
	
	cvReleaseImage( &mask_image );
}

int main( int argc, char** argv )
{
	int selected_image_num = 1;
	char show_ch = 's';
	IplImage* images[NUM_IMAGES];
	vector<IplImage*> template_images;
	IplImage* selected_image = NULL;
	IplImage* temp_image = NULL;
	IplImage* red_point_image = NULL;
	IplImage* connected_reds_image = NULL;
	IplImage* connected_background_image = NULL;
	IplImage* result_image = NULL;
	CvSeq* red_components = NULL;
	CvSeq* background_components = NULL;

	/* the file name of all the templates */
	char templte_image_paths[NUM_TEMPLATE_IMAGES][20] =
	{ 
		"GoRight.JPG", "GoStraight.JPG", "GoLeft.JPG"
		, "NoRightTurn2.JPG", "Parking.JPG", "NoRightTurn.JPG"
		, "NoLeftTurn.JPG", "Yield.JPG", "NoStraight.JPG"
		, "NoParking.JPG"
	};

	// Load all the images.
	for (int file_num=1; (file_num <= NUM_IMAGES); file_num++)
	{
		if( (images[0] = cvLoadImage("./RealRoadSigns.jpg",-1)) == 0 )
			return 0;
		if( (images[1] = cvLoadImage("./RealRoadSigns2.jpg",-1)) == 0 )
			return 0;
		if( (images[2] = cvLoadImage("./ExampleRoadSigns.jpg",-1)) == 0 )
			return 0;
		if( (images[3] = cvLoadImage("./Parking.jpg",-1)) == 0 )
			return 0;
		if( (images[4] = cvLoadImage("./NoParking.jpg",-1)) == 0 )
			return 0;
	}

	/* Load template images, and process template images */
	for(int i=0;i<NUM_TEMPLATE_IMAGES; ++i)
	{
		IplImage *temp_image;
		IplImage *source_image;
		if( (source_image = cvLoadImage(templte_image_paths[i],CV_LOAD_IMAGE_UNCHANGED)) == 0)
			return 0;
		/* store the template image to a vector */
		template_images.push_back(source_image);
		red_point_image = cvCloneImage( template_images[i] );
		result_image = cvCloneImage( template_images[i] );
		temp_image = cvCloneImage( template_images[i] );
		connected_reds_image = cvCloneImage( template_images[i] );
		connected_background_image = cvCloneImage( template_images[i] );

		/* find the red point circle in the template image */
		find_red_points( template_images[i], red_point_image, temp_image );
		/* calculate the connected component of the template image(extract the road sign) */
		red_components = connected_components( red_point_image, connected_reds_image );
		/* flip the foreground and background */
		invert_image( red_point_image, temp_image );
		/* calculate the connected component of the background */
		background_components = connected_components( temp_image, connected_background_image );
		
		determine_optimal_sign_classification( template_images[i], red_point_image, red_components, background_components, result_image );
		template_images[i] = cvCloneImage(result_image);

		CvRect rect;
		/* find the bounding rectangle of the template road sign */
		find_bounding_rectangle(red_components, rect);

		/* cut the templates according to the bounding rectangle(to make the template matching more specific and more effective) */
		template_images[i] = cut_image(template_images[i], rect);
	}

	// Explain the User Interface
    printf( "Hot keys: \n"
            "\tESC - quit the program\n"
			"\t1 - Real Road Signs (image 1)\n"
			"\t2 - Real Road Signs (image 2)\n"
			"\t3 - Synthetic Road Signs\n"
			"\t4 - Synthetic Parking Road Sign\n"
			"\t5 - Synthetic No Parking Road Sign\n"
			"\tr - Show red points\n"
			"\tc - Show connected red points\n"
			"\th - Show connected holes (non-red points)\n"
			"\ts - Show optimal signs\n"
			);
    
	// Create display windows for images
    cvNamedWindow( "Original", 1 );
    cvNamedWindow( "Processed Image", 1 );

	//cvNamedWindow("Result",1);

	// Setup mouse callback on the original image so that the user can see image values as they move the
	// cursor over the image.
	cvSetMouseCallback( "Original", on_mouse_show_values, 0 );
	window_name_for_on_mouse_show_values="Original";
	image_for_on_mouse_show_values=selected_image;

	int user_clicked_key = 0;
	do {
		// Create images to do the processing in.
		if (red_point_image != NULL)
		{
			cvReleaseImage( &red_point_image );
			cvReleaseImage( &temp_image );
			cvReleaseImage( &connected_reds_image );
			cvReleaseImage( &connected_background_image );
			cvReleaseImage( &result_image );
		}
		selected_image = images[selected_image_num-1];
		red_point_image = cvCloneImage( selected_image );
		result_image = cvCloneImage( selected_image );
		temp_image = cvCloneImage( selected_image );
		connected_reds_image = cvCloneImage( selected_image );
		connected_background_image = cvCloneImage( selected_image );

		// Process image
		image_for_on_mouse_show_values = selected_image;
		find_red_points( selected_image, red_point_image, temp_image );
		red_components = connected_components( red_point_image, connected_reds_image );
		invert_image( red_point_image, temp_image );
		background_components = connected_components( temp_image, connected_background_image );
		determine_optimal_sign_classification( selected_image, red_point_image, red_components, background_components, result_image );

		// Show the original & result
        cvShowImage( "Original", selected_image );

		IplImage * recognition_result = cvCloneImage(selected_image);
		recognition_handler(result_image, red_components, template_images, recognition_result);
		

		do {
			if ((user_clicked_key == 'r') || (user_clicked_key == 'c') || (user_clicked_key == 'h') || (user_clicked_key == 's'))
				show_ch = user_clicked_key;

			switch (show_ch)
			{
			case 'c':
				cvShowImage( "Processed Image", connected_reds_image );
				break;
			case 'h':
				cvShowImage( "Processed Image", connected_background_image );
				break;
			case 'r':
				cvShowImage( "Processed Image", red_point_image );
				break;
			case 's':
			default:
				cvShowImage( "Processed Image", result_image );
				break;
			}
	        user_clicked_key = cvWaitKey(0);

		} while ((!((user_clicked_key >= '1') && (user_clicked_key <= '0'+NUM_IMAGES))) &&
			     ( user_clicked_key != ESC ));
		if ((user_clicked_key >= '1') && (user_clicked_key <= '0'+NUM_IMAGES))
		{
			selected_image_num = user_clicked_key-'0';
		}
	} while ( user_clicked_key != ESC );

    return 1;
}
