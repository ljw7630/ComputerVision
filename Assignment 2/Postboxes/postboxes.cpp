#ifdef _CH_
#pragma package <opencv>
#endif

#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "highgui.h"
#include "../utilities.h"

#define VARIATION_ALLOWED_IN_PIXEL_VALUES 30
#define ALLOWED_MOTION_FOR_MOTION_FREE_IMAGE 1.0
#define NUMBER_OF_POSTBOXES 6
#define MINIMUM_GRADIENT_VALUE 5
int PostboxLocations[NUMBER_OF_POSTBOXES][5] = {
                                {   6,  73,  95, 5, 92 }, {   6,  73,  95, 105, 192 },
                                { 105, 158, 193, 5, 92 }, { 105, 158, 193, 105, 192 },
                                { 204, 245, 292, 5, 92 }, { 204, 245, 292, 105, 192 } };
#define POSTBOX_TOP_ROW 0
#define POSTBOX_TOP_BASE_ROW 1
#define POSTBOX_BOTTOM_ROW 2
#define POSTBOX_LEFT_COLUMN 3
#define POSTBOX_RIGHT_COLUMN 4


void indicate_post_in_box( IplImage* image, int postbox )
{
	write_text_on_image(image,(PostboxLocations[postbox][POSTBOX_TOP_ROW]+PostboxLocations[postbox][POSTBOX_BOTTOM_ROW])/2,PostboxLocations[postbox][POSTBOX_LEFT_COLUMN]+2, "Post in");
	write_text_on_image(image,(PostboxLocations[postbox][POSTBOX_TOP_ROW]+PostboxLocations[postbox][POSTBOX_BOTTOM_ROW])/2+19,PostboxLocations[postbox][POSTBOX_LEFT_COLUMN]+2, "this box");
}

void compute_vertical_edge_image(IplImage* input_image, IplImage* output_image)
{
	// TO-DO:  Compute the partial first derivative edge image in order to locate the vertical edges in the passed image,
	//   and then determine the non-maxima suppressed version of these edges (along each row as the rows can be treated
	//   independently as we are only considering vertical edges). Output the non-maxima suppressed edge image. 
	// Note:   You may need to smooth the image first.

	IplImage * tmp = cvCreateImage( 
		cvGetSize(input_image)
		, IPL_DEPTH_8U
		, 1);
	IplImage * first_derivative_img = cvCreateImage(
		cvGetSize(input_image)
		, IPL_DEPTH_16S
		, 1
		);
	IplImage * non_maxima_suppressed_img = cvCloneImage(tmp);
	cvZero(non_maxima_suppressed_img);
	cvCvtColor(input_image, tmp, CV_BGR2GRAY);
	cvSmooth(tmp, tmp,CV_GAUSSIAN, 3, 3);
	cvSobel(tmp, first_derivative_img, 1, 0, 3);
	cvConvertScaleAbs(first_derivative_img, tmp);
	cvThreshold(tmp, tmp, 220, 0, CV_THRESH_TOZERO);

	// non-maximum suppression
	int width_step = tmp->widthStep;
	int pixel_step = tmp->widthStep/tmp->width;
	
	for(int row = 0; row<tmp->height;++row)
	{
		for(int col = 1; col < tmp->width-1;++col)
		{
			unsigned char* curr_point = GETPIXELPTRMACRO(tmp, col, row, width_step, pixel_step);
			unsigned char* pre_point = GETPIXELPTRMACRO(tmp, col-1, row, width_step, pixel_step);
			unsigned char* post_point = GETPIXELPTRMACRO(tmp, col+1, row, width_step, pixel_step);
			unsigned char* non_maxima_suppressed_img_point = GETPIXELPTRMACRO(non_maxima_suppressed_img, col, row, width_step, pixel_step);
			if( (*curr_point) <= (*pre_point)
				|| (*curr_point) < (*post_point) )
			{
				(*non_maxima_suppressed_img_point) = 0;
			}
			else
			{
				(*non_maxima_suppressed_img_point) = *(curr_point);
			}
		}
	}
	cvThreshold(non_maxima_suppressed_img, non_maxima_suppressed_img, 200, 255, CV_THRESH_BINARY);
	cvCvtColor(non_maxima_suppressed_img, output_image, CV_GRAY2BGR);
}

bool motion_free_frame(IplImage* current_frame, IplImage* previous_frame)
{
	// TO-DO:  Determine the percentage of the frames which have changed (by more than VARIATION_ALLOWED_IN_PIXEL_VALUES)
	//        and return whether that percentage is less than ALLOWED_MOTION_FOR_MOTION_FREE_IMAGE.
	//return true;  // Just to allow the system to compile while the code is missing.

	if(previous_frame == NULL)
		return true;
	IplImage *current_frame_grayscale = cvCreateImage(
		cvGetSize(current_frame)
		, IPL_DEPTH_8U
		, 1);
	IplImage *previous_frame_grayscale = cvCreateImage(
		cvGetSize(previous_frame)
		, IPL_DEPTH_8U
		, 1);
	cvCvtColor(current_frame, current_frame_grayscale, CV_BGR2GRAY);
	cvCvtColor(previous_frame, previous_frame_grayscale, CV_BGR2GRAY);
	IplImage *image = cvCreateImage(
		cvGetSize(current_frame_grayscale)
		,IPL_DEPTH_8U
		, 1);
	IplImage *sub_image = cvCreateImage(
		cvGetSize(current_frame_grayscale)
		,IPL_DEPTH_16S
		, 1);
	cvSub(current_frame_grayscale, previous_frame_grayscale, sub_image);
	cvConvertScaleAbs(sub_image, image);
	int width_step = image->widthStep;
	int pixel_step = image->widthStep/image->width;

	int count = 0;
	for(int row = 0; row<image->height;++row)
	{
		for(int col = 0; col < image->width;++col)
		{
			unsigned char* curr_point = GETPIXELPTRMACRO( image, col, row, width_step, pixel_step );

			if( (*curr_point) > VARIATION_ALLOWED_IN_PIXEL_VALUES)
			{
				count ++ ;
			}
		}
	}
	
	float res = ((float)count * 100)/(image->height * image->width);

	if(res < ALLOWED_MOTION_FOR_MOTION_FREE_IMAGE)
		return true;  
	else
		return false;
}

void check_postboxes(IplImage* input_image, IplImage* labelled_output_image, IplImage* vertical_edge_image )
{
	// TO-DO:  If the input_image is not motion free then do nothing.  Otherwise determine the vertical_edge_image and check
	//        each postbox to see if there is mail (by analysing the vertical edges).  Highlight the edge points used during your
	//        processing.  If there is post in a box indicate that there is on the labelled_output_image.
	static IplImage *previous_frame = NULL;

	bool is_motion_free = motion_free_frame(input_image, previous_frame);
	previous_frame = cvCloneImage(input_image);
	if(!is_motion_free)
	{
		return;
	}
	compute_vertical_edge_image(input_image, vertical_edge_image);

	int width_step = vertical_edge_image->widthStep;
	int pixel_step = vertical_edge_image->widthStep/vertical_edge_image->width;


	cvCopy(input_image, labelled_output_image);

	int edge_threshold = 9;
	for(int i=0;i<6;++i)
	{
		int count = 0;
		int row = PostboxLocations[i][POSTBOX_BOTTOM_ROW];
		int start_col = PostboxLocations[i][POSTBOX_LEFT_COLUMN];
		int end_col = PostboxLocations[i][POSTBOX_RIGHT_COLUMN];
		for(int col = start_col; col < end_col;++col)
		{
			unsigned char* curr_point = GETPIXELPTRMACRO(vertical_edge_image, col, row, width_step, pixel_step);
			if(curr_point[0] != 0)
				count ++;
		}
		if(count < edge_threshold)
		{
			indicate_post_in_box(labelled_output_image, i);
		}
	}
}


int main( int argc, char** argv )
{
    IplImage *current_frame=NULL;
	CvSize size;
	size.height = 300; size.width = 200;
	IplImage *corrected_frame = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	IplImage *labelled_image=NULL;
	IplImage *vertical_edge_image=NULL;
    int user_clicked_key=0;
    
    // Load the video (AVI) file
    CvCapture *capture = cvCaptureFromAVI( "./Postboxes.avi" );
    // Ensure AVI opened properly
    if( !capture )
		return 1;    
    
    // Get Frames Per Second in order to playback the video at the correct speed
    int fps = ( int )cvGetCaptureProperty( capture, CV_CAP_PROP_FPS );
    
	// Explain the User Interface
    printf( "Hot keys: \n"
		    "\tESC - quit the program\n"
            "\tSPACE - pause/resume the video\n");

	CvPoint2D32f from_points[4] = { {3, 6}, {221, 11}, {206, 368}, {18, 373} };
	CvPoint2D32f to_points[4] = { {0, 0}, {200, 0}, {200, 300}, {0, 300} };
	CvMat* warp_matrix = cvCreateMat( 3,3,CV_32FC1 );
	cvGetPerspectiveTransform( from_points, to_points, warp_matrix );

	// Create display windows for images
	cvNamedWindow( "Input video", 0 );
	cvNamedWindow( "Vertical edges", 0 );
    cvNamedWindow( "Results", 0 );

	// Setup mouse callback on the original image so that the user can see image values as they move the
	// cursor over the image.
    cvSetMouseCallback( "Input video", on_mouse_show_values, 0 );
	window_name_for_on_mouse_show_values="Input video";

    while( user_clicked_key != ESC ) {
		// Get current video frame
        current_frame = cvQueryFrame( capture );
		image_for_on_mouse_show_values=current_frame; // Assign image for mouse callback
        if( !current_frame ) // No new frame available
			break;

		cvWarpPerspective( current_frame, corrected_frame, warp_matrix );

		if (labelled_image == NULL)
		{	// The first time around the loop create the image for processing
			labelled_image = cvCloneImage( corrected_frame );
			vertical_edge_image = cvCloneImage( corrected_frame );
		}
		check_postboxes( corrected_frame, labelled_image, vertical_edge_image );

		// Display the current frame and results of processing
        cvShowImage( "Input video", current_frame );
        cvShowImage( "Vertical edges", vertical_edge_image );
        cvShowImage( "Results", labelled_image );
        
        // Wait for the delay between frames
        user_clicked_key = cvWaitKey( 1000 / fps );
		if (user_clicked_key == ' ')
		{
			user_clicked_key = cvWaitKey(0);
		}
	}
    
    /* free memory */
    cvReleaseCapture( &capture );
    cvDestroyWindow( "video" );
 
    return 0;
}
