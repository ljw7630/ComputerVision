#ifdef _CH_
#pragma package <opencv>
#endif

#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include "../utilities.h"


#define NUM_IMAGES 9
#define FIRST_LABEL_ROW_TO_CHECK 390
#define LAST_LABEL_ROW_TO_CHECK 490
#define ROW_STEP_FOR_LABEL_CHECK 20


bool find_label_edges( IplImage* edge_image, IplImage* result_image, int row, int& left_label_column, int& right_label_column )
{
	// TO-DO:  Search for the sides of the labels from both the left and right on "row".  The side of the label is taken	
	//        taken to be the second edge located on that row (the side of the bottle being the first edge).  If the label
	//        are found set the left_label_column and the right_label_column and return true.  Otherwise return false.
	//        The routine should mark the points searched (in yellow), the edges of the bottle (in blue) and the edges of the
	//        label (in red) - all in the result_image.

	int width = edge_image->width;

	// get width step and pixel step for grayscale image, used in later "GETPIXELPTRMACRO"
	int gray_width_step = edge_image->widthStep;
	int gray_pixel_step = edge_image->widthStep/width;

	// get width step and pixel step for grayscale image, used in later "PUTPIXELMACRO"
	int rgb_width_step = result_image->widthStep;
	int rgb_pixel_step = result_image->widthStep/result_image->width;

	// scanning from left to right using left_index, scanning from right to left using right_index
	int left_index = 0, right_index = width-1;

	// find_label_edge will stop with left_index greater than right_index, 
	// the count is for counting how many edges have been came across
	int left_count = 0, right_count = 0;

	// initialize the left_label_column and right_label_column
	left_label_column = -1;
	right_label_column = -1;

	// mark the line yellow, the bottle edge blue, the label edge red
	unsigned char yellow_pixel[4] = {0,255,255,0};
	unsigned char blue_pixel[4] = {255,0,0,0};
	unsigned char red_pixel[4] = {0,0,255,0};

	// stop when scanning from left to right meet the scanning from right to left
	while(left_index <= right_index)
	{
		// get the current pixel to scan
		unsigned char* curr_point1 = GETPIXELPTRMACRO(edge_image, left_index, row, gray_width_step, gray_pixel_step);
		unsigned char* curr_point2 = GETPIXELPTRMACRO(edge_image, right_index, row, gray_width_step, gray_pixel_step);

		// draw the current pixel yellow, regardless it's edge point or not(fix the color later)
		// but stop when find the label edge
		if(2 > left_count)
			PUTPIXELMACRO(result_image, left_index, row, yellow_pixel, rgb_width_step, rgb_pixel_step, result_image->nChannels);
		if(2 > right_count)
			PUTPIXELMACRO(result_image, right_index, row, yellow_pixel, rgb_width_step, rgb_pixel_step, result_image->nChannels);

		// if the current point is edge
		if(255 == curr_point1[0])
		{
			// increase the edge count for left scanning
			++left_count;

			// the first edge, it's bottle
			if(1 == left_count)
			{
				PUTPIXELMACRO(result_image, left_index, row, blue_pixel, rgb_width_step, rgb_pixel_step, result_image->nChannels);
			}

			// the second edge, it's label
			if(2 == left_count && -1 == left_label_column)
			{
				// draw the edge red
				PUTPIXELMACRO(result_image, left_index, row, red_pixel, rgb_width_step, rgb_pixel_step, result_image->nChannels);

				// set the column to return
				left_label_column = left_index;
			}
		}

		// if the current point is edge
		if(255 == curr_point2[0])
		{
			// increase the edge count for left scanning
			++right_count;

			// the first edge, it's bottle
			if(1 == right_count)
			{
				PUTPIXELMACRO(result_image, right_index, row, blue_pixel, rgb_width_step, rgb_pixel_step, result_image->nChannels);
			}

			// the second edge, it's label
			if(2 == right_count && -1 == right_label_column)
			{
				// draw the edge red
				PUTPIXELMACRO(result_image, right_index, row, red_pixel, rgb_width_step, rgb_pixel_step, result_image->nChannels);

				// set the column to return
				right_label_column = right_index;
			}
		}
		
		// scan from left to right
		++left_index;
		// scan from right to left
		--right_index;

		// if two edges in the left, two edges in the right, label is presented.
		if(left_count>=2 && right_count>=2)
			return true;
	}

	// if not return before, the label is not present
	return false; 
}

void check_glue_bottle( IplImage* original_image, IplImage* result_image )
{
	// TO-DO:  Inspect the image of the glue bottle passed.  This routine should check a number of rows as specified by 
	//        FIRST_LABEL_ROW_TO_CHECK, LAST_LABEL_ROW_TO_CHECK and ROW_STEP_FOR_LABEL_CHECK.  If any of these searches
	//        fail then "No Label" should be written on the result image.  Otherwise if all left and right column values
	//        are roughly the same "Label Present" should be written on the result image.  Otherwise "Label crooked" should
	//        be written on the result image.

	//         To implement this you may need to use smoothing (cv::GaussianBlur() perhaps) and edge detection (cvCanny() perhaps).
	//        You might also need cvConvertImage() which converts between different types of image.

	// temporay image to store original image
	IplImage * temp_image = cvCloneImage(original_image);

	// grayscale image
	IplImage * gray_image = cvCreateImage(cvGetSize(original_image), IPL_DEPTH_8U, 1);

	// edge image to be used in "find_label_edges"
	IplImage * edge_image = cvCreateImage(cvGetSize(original_image), IPL_DEPTH_8U, 1);
	
	// convert the current image to grayscale
	cvCvtColor(temp_image, gray_image, CV_BGR2GRAY);

	// smooth the image using gaussian blur
	cvSmooth(gray_image, gray_image, CV_GAUSSIAN, 9,9);

	// use canny edge detector to detect edge inside the image
	cvCanny(gray_image, edge_image,45,100);
	
	// get the column of the left and right label in "find_label_edges"
	int left_label_column, right_label_column;

	// previous column of the left and right label. to decide whether the label is crooked
	int pre_left_label_column = -1, pre_right_label_column = -1;

	// if the change of pixel of the label edge point is less than 3, we say the label is not crooked
	int threshold_edge_shift = 3;

	// convert the binary edge image to RGB result image
	cvCvtColor(edge_image, result_image, CV_GRAY2BGR);

	// according to instruction, check the image row by row with step "ROW_STEP_FOR_LABEL_CHECK"
	for(int i=FIRST_LABEL_ROW_TO_CHECK;i<=LAST_LABEL_ROW_TO_CHECK;i+=ROW_STEP_FOR_LABEL_CHECK)
	{
		// see if label is present
		if(find_label_edges(edge_image, result_image, i, left_label_column, right_label_column))
		{
			// we use the first column of the label edge as benchmark to decide whether the label is crooked
			if(-1 == pre_left_label_column)
			{
				pre_left_label_column = left_label_column;
				pre_right_label_column = right_label_column;
			}
			else
			{
				// if the column of the edge is far away from the benchmark by a threshold
				if( (  abs(pre_left_label_column - left_label_column) > threshold_edge_shift)
					|| ( abs(pre_right_label_column - right_label_column) > threshold_edge_shift) )
				{
					// then the label is crooked, write text and return
					write_text_on_image(result_image, 0, 0, "Label crooked");
					return;
				}
			}
		}
		else  // if label is not present, write text "No Label" and return
		{
			write_text_on_image(result_image, 0, 0, "No Label");
			return;
		}
	}

	// if not return before, everything is fine, write "Label present" and return;
	write_text_on_image(result_image, 0, 0, "Label Present");
}

int main( int argc, char** argv )
{
	int selected_image_num = 1;
	IplImage* selected_image = NULL;
	IplImage* images[NUM_IMAGES];
	IplImage* result_image = NULL;

	// Load all the images.
	for (int file_num=1; (file_num <= NUM_IMAGES); file_num++)
	{
		char filename[100];
		sprintf(filename,"./Glue%d.jpg",file_num);
		if( (images[file_num-1] = cvLoadImage(filename,-1)) == 0 )
			return 0;
	}

	// Explain the User Interface
    printf( "Hot keys: \n"
            "\tESC - quit the program\n");
    printf( "\t1..%d - select image\n",NUM_IMAGES);
    
	// Create display windows for images
    cvNamedWindow( "Original", 1 );
    cvNamedWindow( "Processed Image", 1 );

	// Create images to do the processing in.
	selected_image = cvCloneImage( images[selected_image_num-1] );
    result_image = cvCloneImage( selected_image );

	// Setup mouse callback on the original image so that the user can see image values as they move the
	// cursor over the image.
    cvSetMouseCallback( "Original", on_mouse_show_values, 0 );
	window_name_for_on_mouse_show_values="Original";
	image_for_on_mouse_show_values=selected_image;

	int user_clicked_key = 0;
	do {
		// Process image (i.e. setup and find the number of spoons)
		cvCopyImage( images[selected_image_num-1], selected_image );
        cvShowImage( "Original", selected_image );
		image_for_on_mouse_show_values=selected_image;
		check_glue_bottle( selected_image, result_image );
		cvShowImage( "Processed Image", result_image );

		// Wait for user input
        user_clicked_key = cvWaitKey(0);
		if ((user_clicked_key >= '1') && (user_clicked_key <= '0'+NUM_IMAGES))
		{
			selected_image_num = user_clicked_key-'0';
		}
	} while ( user_clicked_key != ESC );

    return 1;
}
