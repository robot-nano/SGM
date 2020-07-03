//
// Created by wserver on 2020/6/21.
//

#include "sgm.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

SGM::SGM(int32 img_height, int32 img_width, int32 window_height, int32 window_width) {
  w_hf_h_ = window_height / 2;
  w_hf_w_ = window_width / 2;
  int feature_size = (img_height - 2 * w_hf_h_) * (img_width - 2 * w_hf_w_);

#if USE_GPU
  // alloc single channel image Tensor
  cudaMalloc(&pImgL_, img_height * img_width * sizeof(uint8));
  cudaMalloc(&pImgR_, img_height * img_width * sizeof(uint8));

  cudaMalloc(&pCensusL_, feature_size * sizeof(uint32));
  cudaMalloc(&pCensusR_, feature_size * sizeof(uint32));
  cudaMalloc(&pCost_, feature_size * MAX_DISPARITY * sizeof(uint8));
  cudaMalloc(&pDisp_, feature_size * sizeof(uint8));
#else
  pCensusL_ = new uint32[feature_size];
  pCensusR_ = new uint32[feature_size];
  pCost_ = new uint8[feature_size * MAX_DISPARITY];
  pDisp_ = new uint8[feature_size];
#endif

  pCensusTransform_ = std::shared_ptr<CensusTransform>(
      new CensusTransform(img_height, img_width, window_height, window_width, pCensusL_, pCensusR_));
  pComputeCost_ = std::shared_ptr<ComputeCost>(
      new ComputeCost(pCost_, pCensusL_, pCensusR_, MAX_DISPARITY,
                      (img_height - 2 * w_hf_h_), (img_width - 2 * w_hf_w_)));
  pComputeDisparity_ = std::shared_ptr<ComputeDisparity>(
      new ComputeDisparity(pDisp_, (img_height - 2 * w_hf_h_), (img_width - 2 * w_hf_w_), pCost_));

}

SGM::~SGM() {
#if USE_GPU
  CudaSafeCall(cudaFree(pImgL_));
  CudaSafeCall(cudaFree(pImgR_));
  CudaSafeCall(cudaFree(pCensusL_));
  CudaSafeCall(cudaFree(pCensusR_));
  CudaSafeCall(cudaFree(pCost_));
  CudaSafeCall(cudaFree(pDisp_));
#else
  delete[] pCensusL_;
  delete[] pCensusR_;
  delete[] pCost_;
  delete[] pDisp_;
#endif
}

void SGM::inference(cv::Mat &img_l, cv::Mat &img_r) {
#if USE_GPU
  int img_size = img_l.rows * img_l.cols;
  cudaMemcpy(pImgL_, (void *) img_l.data, img_size * sizeof(uint8), cudaMemcpyHostToDevice);
  cudaMemcpy(pImgR_, (void *) img_r.data, img_size * sizeof(uint8), cudaMemcpyHostToDevice);
  void *img_left = (void *) pImgL_;
  void *img_right = (void *) pImgR_;
#else
  void *img_left = (void *) (&img_l);
  void *img_right = (void *) (&img_r);
#endif
  pCensusTransform_->census_inference(img_left, img_right);
  pComputeCost_->inference();
  pComputeDisparity_->inference();

  int c_h = img_l.rows - 2 * w_hf_h_;
  int c_w = img_l.cols - 2 * w_hf_w_;

#if USE_GPU
  uint8 *dis = new uint8[c_h * c_w];
  cudaMemcpy(dis, pDisp_, c_h * c_w * sizeof(uint8), cudaMemcpyDeviceToHost);
  cv::Mat disp(c_h, c_w, CV_8UC1, dis);
#else
  cv::Mat disp(c_h, c_w, CV_8UC1, pDisp_);
#endif

  cv::imshow("disp", disp);
  cv::imwrite("../img/disp_gpu.png", disp);
  cv::waitKey(0);
}