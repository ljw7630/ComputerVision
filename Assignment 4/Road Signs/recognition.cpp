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
CvSize get_average_template_size(const vector<IplImage *> templates);
float find_max_value(vector<vector<float> > &result);
float self_implement_calculate_single_matching(const IplImage * src,const IplImage * tmpl, int src_pos_x, int src_pos_y);
void self_implement_template_matching(IplImage * source,const IplImage * templ, vector<vector<float> > &result);
void self_implement_template_matching_handler(const IplImage * src, CvSeq *seq, const vector<IplImage *> templates, IplImage * target);
int count_num_of_pixels(const IplImage * img, CvScalar color);


string int_to_string(const int i)
{
	stringstream out;
	out << i;
	return out.str();
}

IplImage * resize_image(IplImage * source, CvSize size)
{
	IplImage * new_image = cvCreateImage(
		cvSize(size.width, size.height), source->depth, source->nChannels);
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

	if(rect.width < 20 && rect.height < 20)
	{
		rect.width = 20 , rect.height = 20;
	}

	IplImage * scaled_img = resize_image(templ, cvSize(rect.width, rect.height));
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


void draw_bounding_rectangle(IplImage * img, CvRect rect)
{
	cvRectangleR(img, rect, cvScalarAll(255));
}


void recognition_handler(const IplImage *source,CvSeq * seq,const vector<IplImage *> templates, IplImage * target)
{
	// template_matching_handler(source, seq, templates, target);
	self_implement_template_matching_handler(source, seq, templates, target);
}


void template_matching_handler(const IplImage *source,CvSeq * seq,const vector<IplImage *> templates, IplImage * target)
{
	IplImage * source_temp = cvCloneImage(source);
//	IplImage * img = cvCloneImage(source);
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
			scaled_template_img = resize_image(templates[j], cvSize(rects[i].width, rects[i].height));

			//IplImage * result = template_matching(source_temp, scaled_template_img, CV_TM_CCORR_NORMED, &rects[i]);
			IplImage * result = template_matching(source_temp, scaled_template_img, CV_TM_CCOEFF_NORMED, rects[i]);

			double min_val, max_val;
			cvMinMaxLoc(result, &min_val, &max_val);

			// cout << "Index: " << i << " " << max_val << endl;

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
			mark_image(target, templates[result_index], rects[i], 0.4);
			// mark_image(img, templates[result_index], rects[i], 0.3);
		}

		// cout << "Decision: " << result_index << " value: " << maximum_val  << endl << endl;

	}

	cvShowImage("Road signs recognition", target);
	// cvShowImage("Road signs recognition2", img);
}

IplImage * template_matching(IplImage * source,const IplImage * templ, int method, const CvRect const rect)
{
	IplImage * result, * source_temp;

	set_image_ROI(source, rect, 3, 3);
	source_temp = resize_image(source, cvSize(rect.width, rect.height));

	result = cvCreateImage(
		cvSize(source_temp->width - templ->width + 1, source_temp->height - templ->height + 1),
		IPL_DEPTH_32F, 1);
	cvResetImageROI(source);
	
	
	cvMatchTemplate(source_temp, templ, result, method);

	return result;
}

void self_implement_template_matching_handler(const IplImage * src, CvSeq *seq, const vector<IplImage *> templates, IplImage * target)
{
	IplImage * source_temp = cvCloneImage(src);
	//IplImage * img = cvCloneImage(src);
	vector<CvRect> rects;
	int i,j;
	find_bounding_rectangles(seq, rects);

	vector<int> templates_red_pixels(templates.size());
	CvScalar color = cvScalar(0,0,255,0);
	
	foreach_i(i, templates.size())
	{
		templates_red_pixels[i] = count_num_of_pixels(templates[i],color);
	}

	// const double match_threshold = 0.685f;
	const double match_threshold = 0.53f;
	CvSize targetSize = get_average_template_size(templates);
	foreach_i(i, rects.size())
	{
		set_image_ROI(source_temp, rects[i], 3,3);
		//IplImage * temp;
		//cvCopy(source_temp, temp);
		IplImage * scaled_src = resize_image(source_temp, cvSize(targetSize.width+9, targetSize.height+9));
		cvResetImageROI(source_temp);

		float max_val = -1.0;
		int index;
		foreach_i(j,templates.size())
		{
			int result_width = scaled_src->width - templates[j]->width + 1;
			int result_height = scaled_src->height - templates[j]->height + 1;
			vector<vector<float>  > result(result_height, vector<float>(result_width, 0));
			self_implement_template_matching(scaled_src, templates[j], result);
			float max_temp = find_max_value(result);
			if(  max_temp > max_val )
			{
				max_val = max_temp;
				index = j;
			}

			cout << rects[i].x << " " << rects[i].y << " " << max_temp << endl;
		}

		if(max_val > match_threshold)
		{
			int num = count_num_of_pixels(scaled_src, color);
			cout << "src number of red pixels: " << num << " template number of red pixels: " << templates_red_pixels[index] << endl;
			if( abs(num - templates_red_pixels[index]) < 1800)
			{
				cout << "final decision: " << index << " value: " << max_val << endl;
				mark_image(target, templates[index], rects[i], 0.4);
			}
		}

		cout << endl;
	}
	cvShowImage("Road signs recognition", target);
}

CvSize get_average_template_size(const vector<IplImage *> templates)
{
	int width=0, height=0, i;
	foreach_i(i, templates.size())
	{
		width += templates[i]->width;
		height += templates[i]->height;
	}

	return cvSize(width/templates.size(), height/templates.size());
}

// color in BGR
int count_num_of_pixels(const IplImage * img, CvScalar color)
{
	int row, col;
	int pixel_step = img->widthStep/img->width;
	int count = 0;
	foreach_row(row, img)
	{
		foreach_col(col, img)
		{
			unsigned char * point = GETPIXELPTRMACRO(img, col, row, img->width, pixel_step);

			if(point[0] == color.val[0] && point[1] == color.val[1] && point[2] == color.val[2])
			{
				count ++;
			}
		}
	}

	return count;
}

float find_max_value(vector<vector<float> > &result)
{
	float v = -1.0;
	int i,j;
	foreach_i(i, result.size())
	{
		foreach_i(j, result[0].size())
		{
			if(result[i][j] > v)
			{
				v = result[i][j];
			}
		}
	}
	return v;
}

void self_implement_template_matching(IplImage * source,const IplImage * templ, vector<vector<float> > &result)
{
	int x, y;
	foreach_i(x, source->width-templ->width+1)
	{
		foreach_i(y, source->height-templ->height+1)
		{
			result[y][x] = self_implement_calculate_single_matching(source, templ, x, y);
		}
	}
}

float self_implement_calculate_single_matching(const IplImage * src,const IplImage * tmpl, int src_pos_x, int src_pos_y)
{
	int channel, row, col, area;
	float src_tmpl=0, src_square=0, tmpl_square=0;
	// int src_pixel_step = src->widthStep/src->width;
	// int tmpl_pixel_step = tmpl->widthStep/tmpl->width;
	float g_prime, t_prime,g_minus=0, t_minus=0;
	area = tmpl->width * tmpl->height;
	IplImage * gray_img_src = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage * gray_img_tmpl = cvCreateImage(cvGetSize(tmpl), IPL_DEPTH_8U, 1);
	cvCvtColor(src, gray_img_src, CV_BGR2GRAY);
	cvCvtColor(tmpl, gray_img_tmpl, CV_BGR2GRAY);
	int src_pixel_step = gray_img_src->widthStep/gray_img_src->width;
	int tmpl_pixel_step = gray_img_tmpl->widthStep/gray_img_tmpl->width;
	
	foreach_row(row, tmpl)
	{
		foreach_col(col, tmpl)
		{
			unsigned char * g = GETPIXELPTRMACRO(gray_img_src, col+src_pos_x, row+src_pos_y, gray_img_src->widthStep, src_pixel_step);
			unsigned char * t = GETPIXELPTRMACRO(gray_img_tmpl, col, row, gray_img_tmpl->widthStep, tmpl_pixel_step);
			g_minus += (*g);
			t_minus += (*t);
		}
	}
	g_minus = 1.0 / area * g_minus;
	t_minus = 1.0 / area * t_minus;


	foreach_row(row, tmpl)
	{
		foreach_col(col, tmpl)
		{
			g_prime = 0.0;
			t_prime = 0.0;
						
			unsigned char * g = GETPIXELPTRMACRO(gray_img_src, col+src_pos_x, row+src_pos_y, gray_img_src->widthStep, src_pixel_step);
			unsigned char * t = GETPIXELPTRMACRO(gray_img_tmpl, col, row, gray_img_tmpl->widthStep, tmpl_pixel_step);
			g_prime = *g - g_minus;
			t_prime = *t - t_minus;
				
			// src_tmpl += g[channel] * t[channel];
			// src_square += g[channel] * t[channel];
			// tmpl_square += t[channel] * t[channel];

			
			src_tmpl += g_prime * t_prime;
			src_square += g_prime * g_prime;
			tmpl_square += t_prime * t_prime;
		}
	}
	return src_tmpl / sqrtf( src_square*tmpl_square );
	// return (float) src_tmpl / ( sqrtf((float)src_square) * ( sqrtf((float)tmpl_square)) );
	

	/*
	foreach_row(row, tmpl)
	{
		foreach_col(col, tmpl)
		{
			foreach_i(channel,src->nChannels)
			{
				unsigned char * g = GETPIXELPTRMACRO(src, col+src_pos_x, + row+src_pos_y, src->widthStep, src_pixel_step);
				unsigned char * t = GETPIXELPTRMACRO(tmpl, col, row, tmpl->widthStep, tmpl_pixel_step);
				g_minus += g[channel];
				t_minus += t[channel];
			}
		}
	}
	g_minus = 1.0 / area * g_minus;
	t_minus = 1.0 / area * t_minus;


	foreach_row(row, tmpl)
	{
		foreach_col(col, tmpl)
		{
			g_prime = 0.0;
			t_prime = 0.0;
			foreach_i(channel,src->nChannels)
			{
				unsigned char * g = GETPIXELPTRMACRO(src, col+src_pos_x, + row+src_pos_y, src->widthStep, src_pixel_step);
				unsigned char * t = GETPIXELPTRMACRO(tmpl, col, row, tmpl->widthStep, tmpl_pixel_step);
				g_prime += g[channel] - g_minus;
				t_prime += t[channel] - t_minus;
				
				// src_tmpl += g[channel] * t[channel];
				// src_square += g[channel] * t[channel];
				// tmpl_square += t[channel] * t[channel];

			}
			src_tmpl += g_prime * t_prime;
			src_square += g_prime * g_prime;
			tmpl_square += t_prime * t_prime;
		}
	}
	return src_tmpl / sqrtf( src_square*tmpl_square );
	// return (float) src_tmpl / ( sqrtf((float)src_square) * ( sqrtf((float)tmpl_square)) );
	*/
}