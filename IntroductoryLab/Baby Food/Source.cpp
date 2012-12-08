#ifdef _CH_
#pragma package <opencv>
#endif

#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "highgui.h"
#include "../utilities.h"

using namespace cv;

// This routine creates a binary result image where the points are 255,255,255 when the corresponding
// source points are grey/white.  The rule for deciding which points are white/grey is very debatable.
// Should the minimum value be greater?  Should the ratio of max to min values in the point be allowed
// to vary more (or less)?
void select_white_points( IplImage* source, IplImage* result )
{
}

int main( int argc, char** argv )
{
//cvNamedWindow( "Original", 1 );
char filename[100];
sprintf(filename, "../Road Markings/StayingInLane.avi");
std::cout << "hello " << filename << std::endl; 
    CvCapture* VideoCapture = cvCaptureFromAVI( filename );
double NbFrame = cvGetCaptureProperty(VideoCapture, CV_CAP_PROP_FRAME_COUNT);

std::cout << NbFrame << std::endl; 

int user_clicked_key = 0;
do {

// Wait for user input
        user_clicked_key = cvWaitKey(0);


} while ( user_clicked_key != ESC );



return 0;
}