#include "final_convolution.h"

using namespace cv;
using namespace std;

Final_Convolution::Final_Convolution() {
    if_loadKernal = false;
}
Final_Convolution::~Final_Convolution() {}

bool Final_Convolution::load_kernal(Mat kernal) {
    current_kernal = kernal.clone();
    dx = (kernal.cols - 1) / 2;
    dy = (kernal.rows - 1) / 2;
    if_loadKernal = true;
    return true;
}

void Final_Convolution::convolute(const cv::Mat& image, cv::Mat& result, int steps)
{
    if (!if_loadKernal) {
        cout << "kernal is empty!Please load the kernal first!" << endl; return;
    }
    Mat complete_image;
    fillImage(image, complete_image);
    result = Mat::zeros(image.rows, image.cols, image.type());
    int channels = image.channels();
    if (channels == 3) {//���ڶ�δ������ɫͨ����RGBͼƬ���о��
        for (int chan = 0; chan < channels; chan++) {
            for (int i = 0; i < result.rows; i = i + steps) {
                for (int j = 0; j < result.cols; j = j + steps) {
                    computeProduct(i, j, chan, complete_image, result, steps);
                }
            }
        }
        return;
    }
    if (channels == 1) {//���ڶ��ѷ�����ɫͨ���ĵ�ɫͨ�����ͼ��ԭ��ɫͼ���о��
        for (int i = 0; i < result.rows; i=i+steps) {
            for (int j = 0; j < result.cols; j=j+steps) {
                computeProduct(i, j, 0, complete_image, result, steps);
            }
        }
    }
}

void Final_Convolution::computeProduct(int i, int j, int chan, cv::Mat& image, cv::Mat& result, int steps)
{
    if (image.channels() == 3) {
        float sum = 0;
        int drows = i;
        int dcols = j;
        for (int curr_rows = 0; curr_rows < current_kernal.rows; curr_rows=curr_rows+steps) {
            for (int curr_cols = 0; curr_cols < current_kernal.cols; curr_cols=curr_cols+steps) {
                float a = current_kernal.at<float>(curr_rows, curr_cols) * image.at<Vec3b>(curr_rows + drows, curr_cols + dcols)[chan];
                sum += a;
            }
        }
        result.at<Vec3b>(i, j)[chan] = (int)sum;
    }
    else {
        float sum = 0;
        int drows = i;
        int dcols = j;
        for (int curr_rows = 0; curr_rows < current_kernal.rows; curr_rows = curr_rows + steps) {
            for (int curr_cols = 0; curr_cols < current_kernal.cols; curr_cols = curr_cols + steps) {
                float a = current_kernal.at<float>(curr_rows, curr_cols) * image.at<uchar>(curr_rows + drows, curr_cols + dcols);
                sum += a;
            }
        }
        result.at<uchar>(i, j) = (int)sum;
    }
}

void Final_Convolution::fillImage(const cv::Mat& image, cv::Mat& result)
{
    result = Mat::zeros(2 * dy + image.rows, 2 * dx + image.cols, image.type());
    Rect real_roi_of_image = Rect(dx, dy, image.cols, image.rows);
    Mat real_mat_of_image = result(real_roi_of_image);
    image.copyTo(result(real_roi_of_image));
}


//���·�����ɫͨ�������ûҶ��������Ҫֱ�۷�Ӧ��Ӧ��ɫͨ������ɫ�ȣ���������ע�ͽ⿪���ɣ�����ɫͨ������Ϊ��ɫͼƬ������ͬ��
Mat Final_Convolution::split_Bluecolor(cv::Mat target)
{
    int size = target.rows * target.cols * 3;
    Mat b(target.rows, target.cols, CV_8UC1);
    Mat g(target.rows, target.cols, CV_8UC1);
    Mat r(target.rows, target.cols, CV_8UC1);

    Mat out[] = { b, g, r };
    split(target, out);
    /*
    Mat b_color(target.rows, target.cols, CV_8UC3);
    for (int i = 0; i < size; i += 3)
    {
        b_color.data[i] = b.data[i / 3];
        b_color.data[i + 1] = 0;
        b_color.data[i + 2] = 0;
    }
    return b_color;
    */
    return b;
}

Mat Final_Convolution::split_Redcolor(cv::Mat target)
{
    int size = target.rows * target.cols * 3;
    Mat b(target.rows, target.cols, CV_8UC1);
    Mat g(target.rows, target.cols, CV_8UC1);
    Mat r(target.rows, target.cols, CV_8UC1);

    Mat out[] = { b, g, r };
    split(target, out);
    /*
    Mat r_color(target.rows, target.cols, CV_8UC3);
    for (int i = 0; i < size; i += 3)
    {
        r_color.data[i] = 0;
        r_color.data[i + 1] = 0;
        r_color.data[i + 2] = r.data[i / 3];
    }
    return r_color;
    */
    return r;
}

Mat Final_Convolution::split_Greencolor(cv::Mat target)
{
    int size = target.rows * target.cols * 3;
    Mat b(target.rows, target.cols, CV_8UC1);
    Mat g(target.rows, target.cols, CV_8UC1);
    Mat r(target.rows, target.cols, CV_8UC1);

    Mat out[] = { b, g, r };
    split(target, out);
    /*
    Mat g_color(target.rows, target.cols, CV_8UC3);
    for (int i = 0; i < size; i += 3)
    {
        g_color.data[i] = 0;
        g_color.data[i + 1] = g.data[i / 3];
        g_color.data[i + 2] = 0;
    }
    return g_color;
    */
    return g;
}