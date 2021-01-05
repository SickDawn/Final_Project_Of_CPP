#include <iostream>
#include "final_convolution.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat kernal_3 = (Mat_<float>(3, 3) << 0.111, 0.111, 0.111,
	0.111, 0.111, 0.111,
	0.111, 0.111, 0.111);

//5*5��ֵ�����
Mat kernal_5 = (Mat_<float>(5, 5) << 0.04, 0.04, 0.04, 0.04, 0.04,
	0.04, 0.04, 0.04, 0.04, 0.04,
	0.04, 0.04, 0.04, 0.04, 0.04,
	0.04, 0.04, 0.04, 0.04, 0.04,
	0.04, 0.04, 0.04, 0.04, 0.04);


int main()
{
	Final_Convolution projectConvolution;
	Mat image = imread("face.jpg");
	imshow("ԭͼ", image);
	
	Mat resultForKernal_3;
	projectConvolution.load_kernal(kernal_3);
	projectConvolution.convolute(image, resultForKernal_3,1);
	imshow("��3*3�˽��о��", resultForKernal_3);
	

	
	Mat resultForKernal_3_Step2;
	projectConvolution.load_kernal(kernal_3);
	projectConvolution.convolute(image, resultForKernal_3_Step2, 3);
	imshow("��3*3�˽���2�����", resultForKernal_3_Step2);
	
	
	Mat resultForKernal_5;
	projectConvolution.load_kernal(kernal_5);
	projectConvolution.convolute(image, resultForKernal_5, 1);
	imshow("��5*5�˽��о��", resultForKernal_5);
	
	
	Mat imageForRed = projectConvolution.split_Redcolor(image);
	//imshow("��ɫͨ��ԭͼ",imageForRed);
	Mat imageForGreen = projectConvolution.split_Greencolor(image);
	Mat imageForBlue = projectConvolution.split_Bluecolor(image);
	//imshow("��ɫͨ��ԭͼ", imageForBlue);
	
	
	Mat imageForRedConvolution;
	projectConvolution.load_kernal(kernal_3);
	projectConvolution.convolute(imageForRed, imageForRedConvolution, 2);
	imshow("�Ժ�ɫͨ������һ�ξ��", imageForRedConvolution);
	/*
	for (int i = 0; i < 50; i++)
	{
		projectConvolution.convolute(imageForRedConvolution, imageForRedConvolution, 1);
	}
	imshow("�Ժ�ɫͨ������50���", imageForRedConvolution);
	*/
	
	//cout << imageForRedConvolution << endl;

	
	Mat imageForBlueConvolution;
	projectConvolution.load_kernal(kernal_3);
	projectConvolution.convolute(imageForBlue, imageForBlueConvolution, 1);
	imshow("����ɫͨ�����о��", imageForBlueConvolution);
	/*
	for (int i = 0; i < 10; i++)
	{
		projectConvolution.convolute(imageForBlueConvolution, imageForBlueConvolution, 1);
	}
	imshow("����ɫͨ������10�ξ��", imageForBlueConvolution);
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
	imshow("����ɫͨ�����о��", imageForGreenConvolution);
	
	/*
	Mat image2 = imread("face2.jpg");
	imageForRed = projectConvolution.split_Redcolor(image2);
	imageForGreen = projectConvolution.split_Greencolor(image2);
	imageForBlue = projectConvolution.split_Bluecolor(image2);
	
	//�Ժ�ɫͨ�����м���
	Mat resultForRed;
	projectConvolution.load_kernal(imageForRedConvolution);
	projectConvolution.convolute(imageForRed, resultForRed, 1);
	imshow("�Ժ�ɫͨ�����м��", resultForRed);
	*/
	
	imshow("��ɫͨ��", imageForGreen);
	imshow("��ɫͨ��", imageForRed);
	imshow("��ɫͨ��", imageForBlue);
	
	

	waitKey(0);

	return 0;
}