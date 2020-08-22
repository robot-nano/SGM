//
// Created by wserver on 2020/7/24.
//

#include "costs.h"

__global__ void CensusKernelSM2(const uint8_t *im, const uint8_t *im2,
                                cost_t *transform, cost_t *transform2,
                                const uint32_t rows, const uint32_t cols) {
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const int win_rows = (32 + 2 * TOP);
  const int win_cols = (32 + 2 * LEFT);
  __shared__ uint8_t window[win_rows * win_cols];
  __shared__ uint8_t window2[win_rows * win_cols];

  const int id = threadIdx.y * blockDim.x + threadIdx.x;
  const int sm_row = id / win_cols;
  const int sm_col = id % win_cols;

  const int im_row = blockIdx.y * blockDim.y + sm_row - TOP;
  const int im_col = blockIdx.x * blockDim.x + sm_col - LEFT;
  const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
  window[sm_row * win_cols + sm_col] = boundaries ? im[im_row * cols + im_col] : 0;
  window2[sm_row * win_cols + sm_col] = boundaries ? im2[im_row * cols + im_col] : 0;

  const int block_size = blockDim.x * blockDim.y;
  if (id < (win_rows * win_cols - block_size)) {
    const int id = threadIdx.y * blockDim.x + threadIdx.x + block_size;
    const int sm_row = id / win_cols;
    const int sm_col = id % win_cols;

    const int im_row = blockIdx.y * blockDim.y + sm_row - TOP;
    const int im_col = blockIdx.x * blockDim.x + sm_col - LEFT;
    const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
    window[sm_row * win_cols + sm_col] = boundaries ? im[im_row * cols + im_col] : 0;
    window2[sm_row * win_cols + sm_col] = boundaries ? im2[im_row * cols + im_col] : 0;
  }

  __syncthreads();
  uint32_t census = 0;
  uint32_t census2 = 0;
  if (idy < rows && idx < cols) {
    for (int k = 0; k < CENSUS_HEIGHT/2; ++k) {
      for (int m = 0; m < CENSUS_WIDTH; ++m) {
        const uint8_t e1 = window[(threadIdx.y + k) * win_cols + threadIdx.x + m];
        const uint8_t e2 = window[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];
        const uint8_t i1 = window2[(threadIdx.y + k) * win_cols + threadIdx.x + m];
        const uint8_t i2 = window2[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];

        const int shft = k * CENSUS_WIDTH + m;
        uint32_t tmp = (e1 >= e2);
        tmp <<= shft;
        census |= tmp;
        int tmp2 = (i1 >= i2);
        tmp2 <<= shft;
        census2 |= tmp2;
      }
    }
    if (CENSUS_HEIGHT % 2 != 0) {
      const int k = CENSUS_HEIGHT / 2;
      for (int m = 0; m < CENSUS_WIDTH/2; ++m) {
        const uint8_t e1 = window[(threadIdx.y + k) * win_cols + threadIdx.x + m];
        const uint8_t e2 = window[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];
        const uint8_t i1 = window2[(threadIdx.y + k) * win_cols + threadIdx.x + m];
        const uint8_t i2 = window2[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];

        const int shft = k * CENSUS_WIDTH + m;
        uint32_t tmp = (e1 >= e2);
        tmp <<= shft;
        census |= tmp;
        int tmp2 = (i1 >= i2);
        tmp2 <<= shft;
        census2 |= tmp2;
      }
    }

    transform[idy * cols + idx] = census;
    transform2[idy * cols + idx] = census2;
  }
}