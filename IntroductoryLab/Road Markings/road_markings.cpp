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

// for histogram
CvHistogram *hist;
float range_0[]={0,256};
float* ranges[]={range_0};
int (*getTresholdAlgorithm) (IplImage *grayscaleImg);

void showHistogram(IplImage* source)
{
	int hist_size = 64;
	CvHistogram *hist;
	hist=cvCreateHist (1,&hist_size,CV_HIST_ARRAY,ranges,1);
	IplImage *hist_img=cvCreateImage(cvSize(source->width,source->height),8,1);
	float max_val=0;
	cvCalcHist(&source, hist,0,NULL);
	cvGetMinMaxHistValue(hist,0,&max_val,0,0);
	cvScale(hist->bins,hist->bins,((double)hist_img->height)/max_val,0);
	cvSet(hist_img,cvScalarAll(255),0); //intensity values adjustment
	int bin_w=cvRound((double)hist_img->width/hist_size);
	for(int i=0;i<hist_size;++i)
		cvRectangle(hist_img
			,cvPoint(i*bin_w,hist_img->height)
			,cvPoint(
				(i+1)*bin_w
				,hist_img->height-cvRound(cvGetReal1D(hist->bins,i)))
			,cvScalarAll(0),-1,8,0);
	cvShowImage("Histogram",hist_img);
	//IplImage *hist_img = cvCreateImage(cvSize(300,200),8,1);
	//cvEqualizeHist(source, hist_img
}

const int GRAYLEVEL = 256;
const int MAX_BRIGHTNESS = 255;

void select_white_points( IplImage* source, IplImage* result )
{
	cvZero(result);
	IplImage *temp = cvCloneImage(result);
	cvCvtColor(source, temp, CV_RGB2GRAY);
	cvShowImage("Grayscale Image",temp);
	showHistogram(temp);
	// the threshold is the average value of minimum and maximum
	// the result is totally not good
	// int value = getThreshold(NULL, temp);
	//cvThreshold(temp,result, 90, 255, CV_THRESH_BINARY);
	//cvSmooth(temp, temp, CV_GAUSSIAN);

	//cvThreshold(temp, result, 0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
	
	cvAdaptiveThreshold(temp, result, 255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 85,-16);
	cvMorphologyEx(result, result, NULL, NULL, CV_MOP_ERODE, 1);
	
	return;
}

int main( int argc, char** argv )
{
	char filename[100];

	sprintf(filename, "../Road Markings/StayingInLane.avi");
	CvCapture* capture = cvCaptureFromAVI(filename);

	double framerate = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);

	// create display windows for images
	cvNamedWindow("Original", 1);
	cvNamedWindow("Grayscale Image", 1);
	cvNamedWindow("Processed Image",1);
	cvNamedWindow("Histogram",1);
	cvSetMouseCallback("Original", on_mouse_show_values, 0);
	window_name_for_on_mouse_show_values = "Original";

	IplImage* currFrame = 0;
	IplImage* result;

	while(currFrame = cvQueryFrame(capture))
	{
		// create an empty grayscale image
		result = cvCreateImage(
			cvSize(currFrame->width, currFrame->height)
			, IPL_DEPTH_8U
			,1);

		cvShowImage("Original", currFrame);
		image_for_on_mouse_show_values = currFrame;

		select_white_points(currFrame, result);

		cvShowImage("Processed Image", result);

		cvWaitKey(framerate);
	}


	cvReleaseCapture(&capture);
    return 0;
}