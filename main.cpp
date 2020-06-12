//
// Created by wserver on 2020/5/31.
//

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "config.h"
#include "util.h"

#include "census_transform.h"
#include "compute_cost.h"
#include "compute_disp.h"

#if USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

int main(int argc, char** argv) {
  cv::Mat img_left = cv::imread("/home/wserver/ws/Stereo/SGM/img/im2.png",
                                cv::IMREAD_GRAYSCALE);
  cv::Mat img_right = cv::imread("/home/wserver/ws/Stereo/SGM/img/im6.png",
                                 cv::IMREAD_GRAYSCALE);


  int window_height = atoi(argv[1]);
  int window_width  = atoi(argv[2]);
  int w_hf_h = window_height / 2;
  int w_hf_w = window_width  / 2;

  SGM_CHECK_EQ(img_left.rows, img_right.rows);
  SGM_CHECK_EQ(img_left.cols, img_right.cols);

  int census_size = (img_left.rows - w_hf_h * 2) * (img_left.cols - w_hf_w * 2) * sizeof(uint32);
  uint32 *census_l = new uint32[census_size];
  uint32 *census_r = new uint32[census_size];
  uint8 *cost = new uint8[census_size * MAX_DISPARITY / sizeof(uint32)];
  uint8 *disp = new uint8[census_size / sizeof(uint32)];

  CensusTransform census_transform(window_height, window_width);
  ComputeCost compute_cost(cost, census_l, census_r, MAX_DISPARITY,
                           img_left.rows - w_hf_h * 2, img_left.cols - w_hf_w * 2);
  ComputeDisparity compute_disparity(disp, (img_left.rows - w_hf_h * 2),
                                     (img_right.cols - w_hf_w * 2), cost);

#if USE_GPU
  uint8 *cu_img_l, *cu_img_r;
  uint32 *cu_census_l, *cu_census_r;
  cudaMalloc(&cu_img_l, img_left.rows * img_left.cols * sizeof(uint8));
  cudaMalloc(&cu_img_r, img_left.rows * img_right.cols * sizeof(uint8));
  cudaMalloc(&cu_census_l, census_size);
  cudaMalloc(&cu_census_r, census_size);

  cudaMemcpy(cu_img_l, img_left.data,
             img_left.rows * img_left.cols * sizeof(uint8),
             cudaMemcpyHostToDevice);
  cudaMemcpy(cu_img_r, img_right.data,
             img_right.rows * img_right.cols * sizeof(uint8),
             cudaMemcpyHostToDevice);

  census_transform.inference(cu_census_l, cu_census_r, cu_img_l, cu_img_r, img_left.rows, img_left.cols);
  cudaMemcpy(census_l, cu_census_l, census_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(census_r, cu_census_r, census_size, cudaMemcpyDeviceToHost);
#else
  census_transform.inference(census_l, census_r, (void*)(&img_left), (void*)(&img_right));
  compute_cost.inference();
  compute_disparity.inference();
#endif

  cv::Mat disparity((img_left.rows - w_hf_h * 2), (img_left.cols - w_hf_w * 2), CV_8UC1, disp);

  cv::imshow("disp", disparity);

  cv::waitKey(0);
}