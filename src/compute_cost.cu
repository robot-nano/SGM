//
// Created by wserver on 2020/6/9.
//

#include "compute_cost.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ uint8 Hamming32(const uint32& x, const uint32& y) {
  uint8 dist = 0;
  uint32 val = x ^ y;

  while (val) {
    ++dist;
    val &= val - 1;
  }

  return dist;
}

__global__ void compute_cost_kernel(uint8 *cost, uint32 *census_l, uint32 *census_r,
                                    uint32 height, uint32 width, int32 max_disp) {
  int b_idx_y = blockIdx.y * 16;
  int b_idx_x = blockIdx.x * 16;

  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      int idx_x = b_idx_x + x;
      int idx_y = b_idx_y + y;
      if (idx_x < width && idx_y < height) {
        int img_idx = idx_y * width + idx_x;
        int cost_idx = img_idx * max_disp + threadIdx.x;
        if (idx_x >= threadIdx.x) {
          cost[cost_idx] = Hamming32(census_l[img_idx], census_r[img_idx - threadIdx.x]);
        }

        else
          cost[cost_idx] = UINT8_MAX;
      }
    }
  }
}

ComputeCost::ComputeCost(uint8 *cost,
                         uint32 *census_left, uint32 *census_right,
                         int32 max_disparity,
                         uint32 height, uint32 width)
    : max_disparity_(max_disparity), p_cost_(cost),
      p_census_l_(census_left), p_census_r_(census_right),
      height_(height), width_(width)
{}

void ComputeCost::inference() {
#if USE_GPU
  compute_cost_gpu();
#else
  compute_cost_cpu();
#endif
}

uint16 ComputeCost::Hamming32(const uint32 &x, const uint32 &y) {
  uint32 dist = 0, val = x ^ y;

  while (val) {
    ++dist;
    val &= val - 1;
  }

  return dist;
}

void ComputeCost::compute_cost_gpu() {
  int32 grid_dim_x = (width_  + 16 - 1) / 16;
  int32 grid_dim_y = (height_ + 16 - 1) / 16;

  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(max_disparity_);

  compute_cost_kernel<<<grid_dim, block_dim>>>(p_cost_, p_census_l_, p_census_r_,
                                               height_, width_, max_disparity_);
}

void ComputeCost::compute_cost_cpu() {
  for (int32 i = 0; i < height_; ++i) {
    for (int32 j = 0; j < width_; ++j) {
      const uint32 census_val_l = p_census_l_[i * width_ + j];

      for (int32 d = 0; d < max_disparity_; ++d) {
        uint8 &cost = p_cost_[(i * width_ + j) * max_disparity_ + d];
        if (j - d < 0 || j - d >= width_) {
          cost = UINT8_MAX;
          continue;
        }

        const uint32 census_val_r = p_census_r_[i * width_ + j -d];

        cost = Hamming32(census_val_l, census_val_r);
      }
    }
  }
}