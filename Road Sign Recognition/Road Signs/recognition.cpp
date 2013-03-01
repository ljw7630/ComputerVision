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

/* define some shorthands for "the for loop" */
#define foreach_i(i, len)\
	for(i=0;i<len;++i)
#define foreach_f(v, maximum, step)\
	for(v; v<maximum; v+=step)
#define foreach_row(row, img) \
	for(row=0;row<img->height;++row)
#define foreach_col(col, img) \
	for(col=0;col<img->width;++col)

/* function declaration */
void template_matching_handler(const IplImage *source,CvSeq * seq,const vector<IplImage *> templates, IplImage * target);
IplImage * template_matching(IplImage * source,const IplImage * templ, int method, const CvRect const rect);
CvSize get_average_template_size(const vector<IplImage *> templates);
float find_max_value(vector<vector<float> > &result);
float self_implement_calculate_single_matching(const IplImage * src,const IplImage * tmpl, int src_pos_x, int src_pos_y);
void self_implement_template_matching(IplImage * source,const IplImage * templ, vector<vector<float> > &result);
void self_implement_template_matching_handler(const IplImage * src, CvSeq *seq, const vector<IplImage *> templates, IplImage * target);
int count_num_of_pixels(const IplImage * img, CvScalar color);

/* resize the image to a size defined by CvSize */
IplImage * resize_image(IplImage * source, CvSize size)
{
	IplImage * new_image = cvCreateImage(
		cvSize(size.width, size.height), source->depth, source->nChannels);
	cvResize(source, new_image);
	return new_image;
}

/* set the image region of interest */
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

/* draw the small classified image on original image */
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

/* cut image on a specific rectangle */
IplImage * cut_image(IplImage *source, CvRect rect)
{
	IplImage * new_image =  cvCreateImage(
		cvSize(rect.width, rect.height), source->depth, source->nChannels);
	cvSetImageROI(source, rect);
	cvCopy(source, new_image);
	cvResetImageROI(source);
	return new_image;
}

/* find the bounding rectangle of a connected component */
void find_bounding_rectangle(CvSeq * red_component, CvRect &rect)
{
	rect = cvBoundingRect(red_component);
}

/* use "for loop" to find all bounding rectangles on an image */
void find_bounding_rectangles(CvSeq * red_components, vector<CvRect> & bounding_rects)
{
	for(CvSeq * red_component = red_components; red_component != 0;red_component = red_component->h_next)
	{
		if(red_component->total < 10)
			continue;
		bounding_rects.push_back(cvBoundingRect(red_component));
	}
}

/* just pass to template matching handler */
void recognition_handler(const IplImage *source,CvSeq * seq,const vector<IplImage *> templates, IplImage * target)
{
	self_implement_template_matching_handler(source, seq, templates, target);
}

/* the main method for template matching */
void self_implement_template_matching_handler(const IplImage * src, CvSeq *seq, const vector<IplImage *> templates, IplImage * target)
{
	/* copy the source image to a temporary image */
	IplImage * source_temp = cvCloneImage(src);
	vector<CvRect> rects;
	int i,j;

	/* 
	* find the bounding rectangles on the source image 
	* the seq is the sequence of all connected components in the source image
	*/
	find_bounding_rectangles(seq, rects);

	/* count the number of red pixels in each template images */
	vector<int> templates_red_pixels(templates.size());
	/* because the default color channels are in BGR, and I want to count all the red pixels */
	CvScalar color = cvScalar(0,0,255,0);
	
	foreach_i(i, templates.size())
	{
		templates_red_pixels[i] = count_num_of_pixels(templates[i],color);
	}

	/* set a threshold, if the match coefficient is less than this value, 
	 * then the image will not be consider as the same as our template
	 */
	const double match_threshold = 0.53f;

	/* As we know, the sizes of the template images are almost the same, we can use the average value */
	CvSize targetSize = get_average_template_size(templates);

	/* for each real road sign on the image to be classified */
	foreach_i(i, rects.size())
	{
		/* the rectangle is the bounding rectangle of a road sign in the source image
		 * we set the region of interest slightly bigger than the template image */
		set_image_ROI(source_temp, rects[i], 3,3);

		/* scale the real road sign image (which specified by rects[i]) 
		 * the reason I cut the image using ROI(region of interest) is to improve performance
		 */
		IplImage * scaled_src = resize_image(source_temp, cvSize(targetSize.width+9, targetSize.height+9));
		cvResetImageROI(source_temp);

		/* the variables to store the max match value and the index of the match template */
		float max_val = -1.0;
		int index = -1;

		/* for each template */
		foreach_i(j,templates.size())
		{
			/* the result image width = src width - template width + 1
			 * the result image height = src height - template height + 1
			 */
			int result_width = scaled_src->width - templates[j]->width + 1;
			int result_height = scaled_src->height - templates[j]->height + 1;

			/* initialize the result matrix */
			vector<vector<float>  > result(result_height, vector<float>(result_width, 0));

			/* call the template matching algorithm, get the result */
			self_implement_template_matching(scaled_src, templates[j], result);

			/* get the max match value of the current template */
			float max_temp = find_max_value(result);

			/* if the max value is greater than the existing one, replace it and store the template index */
			if(  max_temp > max_val )
			{
				max_val = max_temp;
				index = j;
			}
		}

		/* if the max match value is greater than the match threshold */
		if(max_val > match_threshold)
		{
			/* calculate the number of red pixels in the scaled real road sign image */
			int num = count_num_of_pixels(scaled_src, color);
			
			/* if the road sign pixel count difference is less than a threshold(1800), 
			 * we can sure the image is classified correctly */ 
			if( abs(num - templates_red_pixels[index]) < 1800)
			{
				/* draw a small classified template image on the source image
				 * on some place define by rects[i] */
				mark_image(target, templates[index], rects[i], 0.4);
			}
		}
	}

	// show the classification result on a window */
	cvShowImage("Road signs classification", target);
}

/* I have pre-knowledge about the template images, 
 * the sizes of the templates are almost the same
 * so to improve the performance, I get the average template size */
CvSize get_average_template_size(const vector<IplImage *> templates)
{
	int width=0, height=0, i;
	foreach_i(i, templates.size())
	{
		width += templates[i]->width;
		height += templates[i]->height;
	}

	/* sum all width and devide by templates.size()
	 * sum all height and devide by templates.size()
	 */
	return cvSize(width/templates.size(), height/templates.size());
}

/* color in BGR, we can count the number of pixels which the color is defined by "color" variable
 * in this case, I only count the red pixels
 */
int count_num_of_pixels(const IplImage * img, CvScalar color)
{
	int row, col;
	int pixel_step = img->widthStep/img->width;
	int count = 0;

	/* for each row of the image */
	foreach_row(row, img)
	{
		/* for each column of the image */
		foreach_col(col, img)
		{
			unsigned char * point = GETPIXELPTRMACRO(img, col, row, img->width, pixel_step);

			/* if the current pixel represent a red point, count + 1 */
			if(point[0] == color.val[0] && point[1] == color.val[1] && point[2] == color.val[2])
			{
				count ++;
			}
		}
	}

	return count;
}

/* find the max match value in the result matrix */
float find_max_value(vector<vector<float> > &result)
{
	float v = -1.0;
	int i,j;
	foreach_i(i, result.size())
	{
		foreach_i(j, result[0].size())
		{
			/* if the result[i][j] is greater than the current value, replace it */
			if(result[i][j] > v)
			{
				v = result[i][j];
			}
		}
	}
	return v;
}

/* the main function of the template matching algorithm */
void self_implement_template_matching(IplImage * source,const IplImage * templ, vector<vector<float> > &result)
{
	int x, y;

	/* for each column */
	foreach_i(x, source->width-templ->width+1)
	{
		/* for each row */
		foreach_i(y, source->height-templ->height+1)
		{
			/* calculate the "normalized correlation coefficient" at position (x,y) */
			result[y][x] = self_implement_calculate_single_matching(source, templ, x, y);
		}
	}
}

/* calculate the "normalized correlation coefficient" at a single position
 * the formula can be found at: http://opencv.willowgarage.com/documentation/c/object_detection.html
 * or can be write in latex as:
 
	R(x,y) = \frac {\sum T'(x',y')*I'(x+x',y+y')}{\sqrt{\sum T'(x',y')^{2} * \sum I'(x+x',y+y')^{2} }}
	\\where \hspace{5 mm} T'(x',y') = T'(x',y') - 1/(w*h)\sum T(x'',y'')
	\\ and  \hspace{5 mm} I'(x+x',y+y') = I(x+x',y+y')-1/(w*h)\sum I(x+x'',y+y'')

 */
float self_implement_calculate_single_matching(const IplImage * src,const IplImage * tmpl, int src_pos_x, int src_pos_y)
{
	int row, col, area;
	float src_tmpl=0, src_square=0, tmpl_square=0;
	float g_minus=0, t_minus=0;

	/* area = w*h */
	area = tmpl->width * tmpl->height;

	/* convert the images to gray scale images */
	IplImage * gray_img_src = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage * gray_img_tmpl = cvCreateImage(cvGetSize(tmpl), IPL_DEPTH_8U, 1);
	cvCvtColor(src, gray_img_src, CV_BGR2GRAY);
	cvCvtColor(tmpl, gray_img_tmpl, CV_BGR2GRAY);

	/* calculate pixel step for gray scale images */
	int src_pixel_step = gray_img_src->widthStep/gray_img_src->width;
	int tmpl_pixel_step = gray_img_tmpl->widthStep/gray_img_tmpl->width;
	
	/* calculate T'(x',y') = T'(x',y') - 1/(w*h)\sum T(x'',y'') */

	/* for each row */
	foreach_row(row, tmpl)
	{
		/* for each column */
		foreach_col(col, tmpl)
		{
			/* get the pixel in (col, row) */
			unsigned char * g = GETPIXELPTRMACRO(gray_img_src, col+src_pos_x, row+src_pos_y, gray_img_src->widthStep, src_pixel_step);
			unsigned char * t = GETPIXELPTRMACRO(gray_img_tmpl, col, row, gray_img_tmpl->widthStep, tmpl_pixel_step);

			/* sum the value for the src image */
			g_minus += (*g);
			/* sum the value for the template image */
			t_minus += (*t);
		}
	}
	g_minus = 1.0f / area * g_minus;
	t_minus = 1.0f / area * t_minus;


	/* calculate R(x,y) = \frac {\sum T'(x',y')*I'(x+x',y+y')}{\sqrt{\sum T'(x',y')^{2} * \sum I'(x+x',y+y')^{2} }} */

	/* for each row */
	foreach_row(row, tmpl)
	{
		/* for each column */
		foreach_col(col, tmpl)
		{
			float g_prime = 0.0, t_prime = 0.0;
			
			/* get the pixel in (col, row) */
			unsigned char * g = GETPIXELPTRMACRO(gray_img_src, col+src_pos_x, row+src_pos_y, gray_img_src->widthStep, src_pixel_step);
			unsigned char * t = GETPIXELPTRMACRO(gray_img_tmpl, col, row, gray_img_tmpl->widthStep, tmpl_pixel_step);
			g_prime = *g - g_minus;
			t_prime = *t - t_minus;
			
			src_tmpl += g_prime * t_prime;
			src_square += g_prime * g_prime;
			tmpl_square += t_prime * t_prime;
		}
	}
	return src_tmpl / sqrtf( src_square*tmpl_square );
}