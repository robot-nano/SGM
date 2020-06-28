//
// Created by wserver on 2020/5/31.
//

#include <opencv2/core/core.hpp>
#include "census_transform.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/highgui/highgui.hpp>

__global__ void census_transform_gpu_kernel(
    uint8 *img, uint32 *result,
    int32 img_rows, int32 img_cols,
    int32 w_hf_h_, int32 w_hf_w_) {
  // blockDim.x = blockDim.y = 16
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int tidy = blockDim.y * blockIdx.y + threadIdx.y;

  if ((tidy < img_rows - 2 * w_hf_h_) &&
      (tidx < img_cols - 2 * w_hf_w_)) {
    int center_idx = (tidy + w_hf_h_) * img_cols + tidx + w_hf_w_;
    int result_idx = tidy * (img_cols - 2 * w_hf_h_) + tidx;

    uint32 census_val = 0u;
    for (int32 i = -w_hf_h_; i <= w_hf_h_; ++i) {
      for (int32 j = -w_hf_w_; j <= w_hf_w_; ++j) {
        census_val <<= 1;
        int idx = (tidy + w_hf_h_ + i) * img_cols + (tidx + w_hf_w_ + j);
        if (img[center_idx] > img[idx])
          census_val += 1;
      }
    }

    result[result_idx] = census_val;
  }
}

CensusTransform::CensusTransform(int32 img_height, int32 img_width,
                                 int window_height, int window_width,
                                 uint32 *census_l, uint32 *census_r)
    : pCensusL_(census_l), pCensusR_(census_r),
      imgHeight_(img_height), imgWidth_(img_width) {
  w_hf_h_ = window_height / 2;
  w_hf_w_ = window_width / 2;
}

CensusTransform::~CensusTransform() {
}

void CensusTransform::census_inference(void *img_left, void *img_right) {
#if USE_GPU
  census_transform_gpu(reinterpret_cast<uint8 *>(img_left),
                       pCensusL_, imgHeight_, imgWidth_);
  census_transform_gpu(reinterpret_cast<uint8 *>(img_right),
                       pCensusR_, imgHeight_, imgWidth_);
#else
  census_transform_cpu(*reinterpret_cast<cv::Mat *>(img_left), pCensusL_);
  census_transform_cpu(*reinterpret_cast<cv::Mat *>(img_right), pCensusR_);
#endif
}

void CensusTransform::census_transform_cpu(cv::Mat &img,
                                           uint32 *t_result) {
  int img_height  = img.rows;
  int img_width   = img.cols;

  for (int32 i = w_hf_h_; i < img_height - w_hf_h_; ++i) {
    for (int32 j = w_hf_w_; j < img_width - w_hf_w_; ++j) {
      // central of window
      const uint8 gray_center = img.at<uint8>(i, j);

      uint32 census_val = 0u;
      for (int32 wh = -w_hf_h_; wh <= w_hf_h_; ++wh) {
        for (int32 ww = -w_hf_w_; ww <= w_hf_w_; ++ww) {
          census_val <<= 1;
          const uint8 gray = img.at<uint8>(i + wh, j + ww);
          if (gray_center > gray)
            census_val += 1;
        }
      }

      t_result[(i - w_hf_h_) * (img_width - 2 * w_hf_w_) + (j - w_hf_w_)] = census_val;
    }
  }
}

void CensusTransform::census_transform_gpu(uint8 *img, uint32 *result,
                                           int32 img_rows, int32 img_cols) {
  int32 grid_dim_x = (img_cols - 2 * w_hf_w_ + 16 - 1) / 16;
  int32 grid_dim_y = (img_rows - 2 * w_hf_h_ + 16 - 1) / 16;
  dim3 gradDim(grid_dim_x, grid_dim_y);
  dim3 blockDim(16, 16);
  census_transform_gpu_kernel<<<gradDim, blockDim>>>(img, result, img_rows, img_cols,
                                                     w_hf_h_, w_hf_w_);
}