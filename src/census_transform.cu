//
// Created by wserver on 2020/5/31.
//

#include <opencv2/core/core.hpp>
#include "census_transform.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__
__launch_bounds__(1024, 2)
void census_transform_gpu_kernel(const uint8 *im, const uint8 *im2,
                                            uint32 *transform, uint32 *transform2,
                                            const uint32 rows, const uint32 cols) {
  const uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32 idy = blockIdx.y * blockDim.y + threadIdx.y;

  const int win_cols = (32 + LEFT * 2); // 32+4*2 = 40
  const int win_rows = (32 + TOP * 2); // 32+3*2 = 38

  __shared__ uint8_t window[win_cols * win_rows];
  __shared__ uint8_t window2[win_cols * win_rows];

  const int id = threadIdx.y * blockDim.x + threadIdx.x;
  const int sm_row = id / win_cols;
  const int sm_col = id % win_cols;

  const int im_row = blockIdx.y * blockDim.y + sm_row - TOP;
  const int im_col = blockIdx.x * blockDim.x + sm_col - LEFT;
  const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
  window2[sm_row * win_cols + sm_col] = boundaries ? im[im_row * cols + im_col] : 0;
  window[sm_row * win_cols + sm_col] = boundaries ? im2[im_row * cols + im_col] : 0;

  // Not enough threads to fill window and window2
  const int block_size = blockDim.x * blockDim.y;
  if (id < (win_cols * win_rows - block_size)) {
    const int id = threadIdx.y * blockDim.x + threadIdx.x + block_size;
    const int sm_row = id / win_cols;
    const int sm_col = id % win_cols;

    const int im_row = blockIdx.y * blockDim.y + sm_row - TOP;
    const int im_col = blockIdx.x * blockDim.x + sm_col - LEFT;
    const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
    window2[sm_row * win_cols + sm_col] = boundaries ? im[im_row * cols + im_col] : 0;
    window[sm_row * win_cols + sm_col] = boundaries ? im2[im_row * cols + im_col] : 0;
  }

  __syncthreads();
  uint32 census = 0;
  uint32 census2 = 0;
  if (idy < rows && idx < cols) {
    for (int k = 0; k < CENSUS_HEIGHT / 2; k++) {
      for (int m = 0; m < CENSUS_WIDTH; m++) {
        const uint8_t e1 = window[(threadIdx.y + k) * win_cols + threadIdx.x + m];
        const uint8_t e2 = window[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];
        const uint8_t i1 = window2[(threadIdx.y + k) * win_cols + threadIdx.x + m];
        const uint8_t i2 = window2[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];

        const int shft = k * CENSUS_WIDTH + m;
        // Compare to the center
        uint32 tmp = (e1 >= e2);
        // Shift to the desired position
        tmp <<= shft;
        // Add it to its place
        census |= tmp;
        // Compare to the center
        uint32 tmp2 = (i1 >= i2);
        // Shift to the desired position
        tmp2 <<= shft;
        // Add it to its place
        census2 |= tmp;
      }
    }
//    if (CENSUS_HEIGHT % 2 != 0) {
//      const int k = CENSUS_HEIGHT / 2;
//      for (int m = 0; m < CENSUS_WIDTH / 2; m++) {
//        const uint8_t e1 = window[(threadIdx.y + k) * win_cols + threadIdx.x + m];
//        const uint8_t e2 = window[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];
//        const uint8_t i1 = window2[(threadIdx.y + k) * win_cols + threadIdx.x + m];
//        const uint8_t i2 = window2[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];
//
//        const int shft = k * CENSUS_WIDTH + m;
//        // Compare to the center
//        uint32 tmp = (e1 >= e2);
//        // Shift to the desired position
//        tmp <<= shft;
//        // Add it to its place
//        census |= tmp;
//        // Compare to the center
//        uint32 tmp2 = (i1 >= i2);
//        // Shift to the desired position
//        tmp2 <<= shft;
//        // Add it to its place
//        census2 |= tmp2;
//      }
//    }
    transform[idy * cols + idx] = census;
    transform2[idy * cols + idx] = census2;
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

void CensusTransform::census_inference(void *img_left, void *img_right, uint32 *census_l, uint32 *census_r) {
#if USE_GPU
  static cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  dim3 block_size;
  block_size.x = 32;
  block_size.y = 32;
  dim3 grid_size;
  grid_size.x = (imgWidth_ + block_size.x - 1) / block_size.x;
  grid_size.y = (imgHeight_ + block_size.y - 1) / block_size.y;

  census_transform_gpu_kernel<<<grid_size, block_size, 0, stream1>>>(
      reinterpret_cast<uint8*>(img_left), reinterpret_cast<uint8*>(img_right),
      census_l, census_r, imgHeight_, imgWidth_);
  cudaStreamSynchronize(stream1);
#else
  census_transform_cpu(*reinterpret_cast<cv::Mat *>(img_left), pCensusL_);
  census_transform_cpu(*reinterpret_cast<cv::Mat *>(img_right), pCensusR_);
#endif
}

void CensusTransform::census_transform_cpu(cv::Mat &img,
                                           uint32 *t_result) {
  int img_height = img.rows;
  int img_width = img.cols;

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