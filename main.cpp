//
// Created by wserver on 2020/5/31.
//

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "census_transform.h"
#include "config.h"
#include "util.h"

#if USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

int main(int argc, char** argv) {
  cv::Mat img_left = cv::imread("/home/wserver/ws/Stereo/SGM/img/left.png",
                                cv::IMREAD_GRAYSCALE);
  cv::Mat img_right = cv::imread("/home/wserver/ws/Stereo/SGM/img/right.png",
                                 cv::IMREAD_GRAYSCALE);


  int window_height = atoi(argv[1]);
  int window_width  = atoi(argv[2]);
  int w_hf_h = window_height / 2;
  int w_hf_w = window_width  / 2;

  SGM_CHECK_EQ(img_left.rows, img_right.rows);
  SGM_CHECK_EQ(img_left.cols, img_right.cols);


  CensusTransform census_transform(window_height, window_width);

  uint32 *census_l = new uint32[(img_left.rows - w_hf_h * 2) * (img_left.cols - w_hf_w * 2) * 4];
  uint32 *census_r = new uint32[(img_right.rows - w_hf_h * 2) * (img_right.cols - w_hf_w * 2) * 4];
#if USE_GPU
    uint8 *cu_img_l, *cu_img_r;
    uint32 *cu_census_l, *cu_census_r;
    cudaMalloc(&cu_img_l, img_left.rows * img_left.cols * sizeof(uint8));
    cudaMalloc(&cu_img_r, img_left.rows * img_right.cols * sizeof(uint8));
    cudaMalloc(&cu_census_l, (img_left.rows - w_hf_h * 2) * (img_left.cols - w_hf_w * 2) * sizeof(uint32));
    cudaMalloc(&cu_census_r, (img_left.rows - w_hf_h * 2) * (img_left.cols - w_hf_w * 2) * sizeof(uint32));

    cudaMemcpy((void*)cu_img_l, (void*)img_left.data,
               img_left.rows * img_left.cols * sizeof(uint8),
               cudaMemcpyHostToDevice);
    cudaMemcpy((void*)cu_img_r, (void*)img_right.data,
               img_right.rows * img_right.cols * sizeof(uint8),
               cudaMemcpyHostToDevice);

    census_transform.inference(cu_census_l, cu_census_r, cu_img_l, cu_img_r);
#else
    census_transform.inference(census_l, census_r, (void*)(&img_left), (void*)(&img_right));
#endif

      uint16 *il = new uint16[(img_left.rows - w_hf_h * 2) * (img_left.cols - w_hf_w * 2)];
      uint16 *ir = new uint16[(img_left.rows - w_hf_h * 2) * (img_left.cols - w_hf_w * 2)];
    for (int i = 0; i < img_left.rows - w_hf_h * 2; ++i) {
      for (int j = 0; j < img_left.cols - w_hf_w * 2; ++j) {
        il[i * (img_left.cols - w_hf_h) + j] = census_l[i * (img_left.cols - w_hf_h) + j];
        ir[i * (img_left.cols - w_hf_h) + j] = census_r[i * (img_left.cols - w_hf_h) + j];
      }
    }

  cv::Mat result_l = cv::Mat(img_left.rows - w_hf_h * 2, img_left.cols - w_hf_w * 2, CV_16UC1, il);
  cv::Mat result_r = cv::Mat(img_right.rows - w_hf_h * 2, img_right.cols - w_hf_w * 2, CV_16UC1, ir);

/*  for (int i = 0; i < result_l.rows; ++i) {
    for (int j = 0; j < result_l.cols; ++j) {
      std::cout << static_cast<int32>(result_l.at<uchar>(j, i)) << std::endl;
    }
  }*/
  cv::imwrite("./result_l.png", result_l);
  cv::imwrite("./result_r.png", result_r);

  cv::waitKey(0);
}