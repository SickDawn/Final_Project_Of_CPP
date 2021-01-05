#pragma once
#ifndef FINAL_CONVOLUTION
#define FINAL_CONVOLUTION

#include <opencv2/opencv.hpp>

class Final_Convolution {
public:
	Final_Convolution();
	~Final_Convolution();
	void convolute(const cv::Mat& image, cv::Mat& result, int steps);
	bool load_kernal(cv::Mat kernal);
	
	cv::Mat split_Redcolor(cv::Mat target);
	cv::Mat split_Greencolor(cv::Mat target);
	cv::Mat split_Bluecolor(cv::Mat target);
	

private:
	bool if_loadKernal;
	cv::Mat current_kernal;
	int dx, dy;

	void computeProduct(int i, int j, int chan, cv::Mat& image, cv::Mat& result, int steps);
	void fillImage(const cv::Mat& image, cv::Mat& result);
};

#endif