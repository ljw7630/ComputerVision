#ifdef _CH_
#pragma package <opencv>
#endif

#include "cv.h"
#include "highgui.h"
#include "../utilities.h"
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <algorithm>
#include <tuple>
#include <sstream>

using namespace std;

void template_matching_handler(const IplImage *source,CvSeq * seq,const vector<IplImage *> templates, IplImage * target);
IplImage * template_matching(IplImage * source,const IplImage * templ, int method, const CvRect const rect);

string int_to_string(const int i)
{
	stringstream out;
	out << i;
	return out.str();
}

IplImage * resize_image(IplImage * source, CvRect rect)
{
	IplImage * new_image = cvCreateImage(
		cvSize(rect.width, rect.height), source->depth, source->nChannels);
	cvResize(source, new_image);
	return new_image;
}

void set_image_ROI(IplImage * source, const CvRect &rect, int expand_width=0, int expand_height = 0)
{
	int x0, y0, x1, y1;

	if( rect.x - expand_width < 0)
		 x0 = 0;
	else
		x0 = rect.x - expand_width;

	if( rect.y - expand_height < 0)
		 y0 = 0;
	else
		y0 = rect.y - expand_height;

	if( x0 + rect.width + 2 * expand_width > source->width)
		x1 = source->width-1;
	else
		x1 = x0 + rect.width + 2 * expand_width;

	if( y0 + rect.height + 2 * expand_height > source->height)
		y1 = source->height-1;
	else
		y1 = y0 + rect.height + 2 * expand_height;

	cvSetImageROI(source, cvRect(x0,y0, x1-x0, y1-y0));
}

void mark_image(IplImage *source, IplImage * templ, CvRect rect, double scale)
{
	rect.width = (int)rect.width * scale;
	rect.height = (int)rect.height * scale;

	IplImage * scaled_img = resize_image(templ, rect);
	cvSetImageROI(source, rect);
	cvCopy(scaled_img, source);
	cvResetImageROI(source);
}

IplImage * cut_image(IplImage *source, CvRect rect)
{
	IplImage * new_image =  cvCreateImage(
		cvSize(rect.width, rect.height), source->depth, source->nChannels);
	cvSetImageROI(source, rect);
	cvCopy(source, new_image);
	cvResetImageROI(source);
	return new_image;
}

void find_bounding_rectangle(CvSeq * red_component, CvRect &rect)
{
	rect = cvBoundingRect(red_component);
}

void find_bounding_rectangles(CvSeq * red_components, vector<CvRect> & bounding_rects)
{
	for(CvSeq * red_component = red_components; red_component != 0;red_component = red_component->h_next)
	{
		if(red_component->total < 10)
			continue;
		bounding_rects.push_back(cvBoundingRect(red_component));
	}
}

/*
double square_distance(int x0, int y0, int x1, int y1)
{
	return sqrt(  pow((double)x0-x1, 2) + pow((double)y0-y1,2) );
}

int counting_black_pixels(IplImage * img, const CvRect * rect = NULL)
{
	double radius;
	int center_x;
	int center_y;
	int width, height;
	int start_x, start_y;
	if(rect != NULL)
	{
		radius = ((double)rect->width / 2 + (double)rect->height/2)/2;
		center_x = rect->width/2;
		center_y = rect->height/2;
		width = rect->width;
		height= rect->height;
		start_x = rect->x;
		start_y = rect->y;
	}
	else
	{
		radius = ((double)img->width/2 + (double)img->height/2) / 2;
		center_x = img->width / 2;
		center_y = img->height /2;
		width = img->width;
		height = img->height;
		start_x = 0;
		start_y = 0;
	}

	int col, row;
	int width_step = img->widthStep;
	int pixel_step = img->widthStep/img->width;
	int counter = 0;
	for(row = 0; row < width; ++row)
	{
		for(col = 0; col < height; ++col)
		{
			if( square_distance(col + start_x, row+ start_y, center_x + start_x, center_y + start_y) < radius)
			{
				unsigned char * p = GETPIXELPTRMACRO( img, col + start_x, row+start_y, width_step, pixel_step );
				if(p[0] == 0 && p[1] == 0 && p[2] == 0)
					counter++;
			}
		}
	}

	return counter;
}

void remove_background(IplImage * src_img, IplImage * dest_img)
{
	int row, col;
	int width_step = src_img->widthStep;
	int pixel_step = src_img->widthStep/src_img->width;
	int threshold = 200;
	unsigned char b_g[4] = {255,255,0,0};
	foreach_row(row, src_img)
	{
		foreach_col(col, src_img)
		{
			unsigned char* curr_point1 = GETPIXELPTRMACRO( src_img, col, row, width_step, pixel_step );
			unsigned char* curr_point2 = GETPIXELPTRMACRO( dest_img, col, row, width_step, pixel_step );

			if( abs(curr_point1[0] - curr_point2[0]) > threshold
				&& abs(curr_point1[1] - curr_point2[1]) > threshold
				&& abs(curr_point1[2] - curr_point2[2]) > threshold)
			{
				PUTPIXELMACRO( dest_img, col, row, b_g, width_step, pixel_step, dest_img->nChannels );
			}
			else if( curr_point2[0] == 0 && curr_point2[1] == 0 && curr_point2[2] == 255 )
			{
				PUTPIXELMACRO( dest_img, col, row, b_g, width_step, pixel_step, dest_img->nChannels );
			}
			else if( curr_point2[0] == 255 && curr_point2[1] == 255 && curr_point2[2] == 255 )
			{
				PUTPIXELMACRO( dest_img, col, row, b_g, width_step, pixel_step, dest_img->nChannels );
			}
		}
	}
}

*/

void draw_bounding_rectangle(IplImage * img, CvRect rect)
{
	cvRectangleR(img, rect, cvScalarAll(255));
}


void recognition_handler(const IplImage *source,CvSeq * seq,const vector<IplImage *> templates, IplImage * target)
{
	template_matching_handler(source, seq, templates, target);
}


void template_matching_handler(const IplImage *source,CvSeq * seq,const vector<IplImage *> templates, IplImage * target)
{
	IplImage * source_temp = cvCloneImage(source);
	IplImage * img = cvCloneImage(source);
	vector<CvRect> rects;
	find_bounding_rectangles(seq, rects);
	
	int i,j;
	const double match_threshold = 0.685;

	foreach_i(i, rects.size())
	{
		// scale the templates to match the size of the rect

		int result_index = -1;
		double maximum_val = -1;
		IplImage * scaled_template_img;
		foreach_i(j, templates.size())
		{
			scaled_template_img = resize_image(templates[j], rects[i]);

			//IplImage * result = template_matching(source_temp, scaled_template_img, CV_TM_CCORR_NORMED, &rects[i]);
			IplImage * result = template_matching(source_temp, scaled_template_img, CV_TM_CCOEFF_NORMED, rects[i]);

			double min_val, max_val;
			cvMinMaxLoc(result, &min_val, &max_val);

			cout << "Index: " << i << " " << max_val << endl;

			if(max_val < match_threshold)
				continue;

			if(maximum_val < max_val)
			{
				maximum_val  = max_val;
				result_index = j;
			}
			
		}

		if(match_threshold < maximum_val)
		{
			mark_image(target, templates[result_index], rects[i], 0.3);
			mark_image(img, templates[result_index], rects[i], 0.3);
		}

		cout << "Decision: " << result_index << " value: " << maximum_val  << endl << endl;

	}

	cvShowImage("Road signs recognition1", target);
	cvShowImage("Road signs recognition2", img);
}

IplImage * template_matching(IplImage * source,const IplImage * templ, int method, const CvRect const rect)
{
	IplImage * result, * source_temp;

	set_image_ROI(source, rect, 3, 3);
	source_temp = resize_image(source, cvRect(rect.x, rect.y, rect.width + 9, rect.height + 9));

	result = cvCreateImage(
		cvSize(source_temp->width - templ->width + 1, source_temp->height - templ->height + 1),
		IPL_DEPTH_32F, 1);
	cvResetImageROI(source);
	
	
	cvMatchTemplate(source_temp, templ, result, method);

	return result;
}