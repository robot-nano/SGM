//
// Created by wserver on 2020/6/11.
//

#include "compute_disp.h"
#include <algorithm>

__device__ uint8 find_min_128(uint8 *cost) {
  for (int i = 1; i < 4; ++i) {
    cost[threadIdx.x] = min(cost[threadIdx.x], cost[threadIdx.x + i * threadIdx.x]);
  }
  for (int offset = 16; offset >= 1; offset /= 2) {
    cost[threadIdx.x] = min(cost[threadIdx.x], __shfl_down_sync(MASK, cost[threadIdx.x], offset, 32));
  }
  uint8 min = UINT8_MAX;
  if (threadIdx.x == 0)
    min = cost[threadIdx.x];
  return min;
}

__global__ void compute_disp_kernel(uint8 *disparity,
                                    uint32 height, uint32 width,
                                    uint8 *cost) {
  int idx_x = blockIdx.x * 4 + threadIdx.x;
  int idx_y = blockIdx.y * 4 + threadIdx.y;

  if (idx_y < height && idx_x < width) {
    int idx = idx_y * width + idx_x;
    if (idx < height * width) {
      cost[idx] = find_min_128(disparity + idx * 128);
    }
  }
}

ComputeDisparity::ComputeDisparity(uint8 *disparity,
                                   uint32 height, uint32 width,
                                   uint8 *cost)
    : height_(height), width_(width),
      pDisparity_(disparity), pCost_(cost) {}

void ComputeDisparity::inference() {
#if USE_GPU
  compute_disparity_gpu();
#else
  compute_disparity_cpu();
#endif
}

void ComputeDisparity::compute_disparity_gpu() {
  int grid_dim_y = (height_ + 4 - 1) / 4;
  int grid_dim_x = (width_  + 4 - 1) / 4;

  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(4, 4);
  compute_disp_kernel<<<grid_dim, block_dim>>>(pDisparity_, height_, width_, pCost_);
}

void ComputeDisparity::compute_disparity_cpu() {
  uint8 *cost_ptr = pCost_;

  for (int32 i = 0; i < height_; ++i) {
    for (int32 j = 0; j < width_; ++j) {
      uint8 min_cost = UINT8_MAX;
      uint8 max_cost = 0;
      int32 best_disparity = 0;

      for (int32 d = 0; d < MAX_DISPARITY; ++d) {
        const uint8 cost = cost_ptr[(i * width_ + j) * MAX_DISPARITY + d];
        if (min_cost > cost) {
          min_cost = cost;
          best_disparity = d;
        }
        max_cost = std::max(max_cost, cost);
      }

      if (max_cost != min_cost) {
        pDisparity_[i * width_ + j] = best_disparity;
      } else {
        pDisparity_[i * width_ + j] = UINT8_MAX;
      }
    }
  }
}