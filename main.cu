//
// Created by wserver on 2020/5/31.
//


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "sgm.h"

int main(int argc, char **argv) {
  int window_height = atoi(argv[1]);
  int window_width = atoi(argv[2]);

  cv::Mat img_left = cv::imread("/home/wserver/ws/Stereo/SGM/img/im2.png",
                                cv::IMREAD_GRAYSCALE);
  cv::Mat img_right = cv::imread("/home/wserver/ws/Stereo/SGM/img/im6.png",
                                 cv::IMREAD_GRAYSCALE);
  int rows = img_left.rows / 32 * 32;
  int cols = img_left.cols / 32 * 32;
  cv::resize(img_left, img_left, cv::Size(cols, rows));
  cv::resize(img_right, img_right, cv::Size(cols, rows));
  SGM sgm(img_left.rows, img_left.cols, window_height, window_width);
  sgm.inference(img_left, img_right);
}