//
// Created by wserver on 2020/5/31.
//


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "sgm.h"

int main(int argc, char **argv) {
  int window_height = atoi(argv[1]);
  int window_width = atoi(argv[2]);

  cv::Mat img_left = cv::imread("/home/wserver/ws/Stereo/SGM/img/left.png",
                                cv::IMREAD_GRAYSCALE);
  cv::Mat img_right = cv::imread("/home/wserver/ws/Stereo/SGM/img/right.png",
                                 cv::IMREAD_GRAYSCALE);
  SGM sgm(img_left.rows, img_left.cols, window_height, window_width);
  sgm.inference(img_left, img_right);
}