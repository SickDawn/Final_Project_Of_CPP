#include <iostream>
#include "final_convolution.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat kernal_3 = (Mat_<float>(3, 3) << 0.111, 0.111, 0.111,
	0.111, 0.111, 0.111,
	0.111, 0.111, 0.111);

//5*5均值卷积核
Mat kernal_5 = (Mat_<float>(5, 5) << 0.04, 0.04, 0.04, 0.04, 0.04,
	0.04, 0.04, 0.04, 0.04, 0.04,
	0.04, 0.04, 0.04, 0.04, 0.04,
	0.04, 0.04, 0.04, 0.04, 0.04,
	0.04, 0.04, 0.04, 0.04, 0.04);


int main()
{
	Final_Convolution projectConvolution;
	Mat image = imread("face.jpg");
	imshow("原图", image);
	
	Mat resultForKernal_3;
	projectConvolution.load_kernal(kernal_3);
	projectConvolution.convolute(image, resultForKernal_3,1);
	imshow("用3*3核进行卷积", resultForKernal_3);
	

	
	Mat resultForKernal_3_Step2;
	projectConvolution.load_kernal(kernal_3);
	projectConvolution.convolute(image, resultForKernal_3_Step2, 3);
	imshow("用3*3核进行2步卷积", resultForKernal_3_Step2);
	
	
	Mat resultForKernal_5;
	projectConvolution.load_kernal(kernal_5);
	projectConvolution.convolute(image, resultForKernal_5, 1);
	imshow("用5*5核进行卷积", resultForKernal_5);
	
	
	Mat imageForRed = projectConvolution.split_Redcolor(image);
	//imshow("红色通道原图",imageForRed);
	Mat imageForGreen = projectConvolution.split_Greencolor(image);
	Mat imageForBlue = projectConvolution.split_Bluecolor(image);
	//imshow("蓝色通道原图", imageForBlue);
	
	
	Mat imageForRedConvolution;
	projectConvolution.load_kernal(kernal_3);
	projectConvolution.convolute(imageForRed, imageForRedConvolution, 2);
	imshow("对红色通道进行一次卷积", imageForRedConvolution);
	/*
	for (int i = 0; i < 50; i++)
	{
		projectConvolution.convolute(imageForRedConvolution, imageForRedConvolution, 1);
	}
	imshow("对红色通道进行50卷积", imageForRedConvolution);
	*/
	
	//cout << imageForRedConvolution << endl;

	
	Mat imageForBlueConvolution;
	projectConvolution.load_kernal(kernal_3);
	projectConvolution.convolute(imageForBlue, imageForBlueConvolution, 1);
	imshow("对蓝色通道进行卷积", imageForBlueConvolution);
	/*
	for (int i = 0; i < 10; i++)
	{
		projectConvolution.convolute(imageForBlueConvolution, imageForBlueConvolution, 1);
	}
	imshow("对蓝色通道进行10次卷积", imageForBlueConvolution);
	*/
	
	
	Mat imageForGreenConvolution;
	projectConvolution.load_kernal(kernal_3);
	projectConvolution.convolute(imageForGreen, imageForGreenConvolution, 1);
	/*
	for (int i = 0; i < 10; i++)
	{
		projectConvolution.convolute(imageForGreenConvolution, imageForGreenConvolution, 1);
	}
	*/
	imshow("对绿色通道进行卷积", imageForGreenConvolution);
	
	/*
	Mat image2 = imread("face2.jpg");
	imageForRed = projectConvolution.split_Redcolor(image2);
	imageForGreen = projectConvolution.split_Greencolor(image2);
	imageForBlue = projectConvolution.split_Bluecolor(image2);
	
	//对红色通道进行鉴别
	Mat resultForRed;
	projectConvolution.load_kernal(imageForRedConvolution);
	projectConvolution.convolute(imageForRed, resultForRed, 1);
	imshow("对红色通道进行检测", resultForRed);
	*/
	
	imshow("绿色通道", imageForGreen);
	imshow("红色通道", imageForRed);
	imshow("蓝色通道", imageForBlue);
	
	

	waitKey(0);

	return 0;
}