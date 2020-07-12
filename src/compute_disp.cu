//
// Created by wserver on 2020/6/11.
//

#include "compute_disp.h"
#include <algorithm>

__device__ void find_min_128(uint8 *cost, uint32 height, uint32 width) {
  unsigned int idx_x = blockIdx.x * 4 + threadIdx.x;
  unsigned int idx_y = blockIdx.y * 4 + threadIdx.y;
  unsigned int idx = (idx_y * width + idx_x) * 128;

  for (int i = 1; i < 4; ++i) {
    cost[idx + threadIdx.z] = min(cost[idx + threadIdx.z], cost[idx + threadIdx.z + i * blockDim.z]);
  }
  for (int offset = 16; offset >= 1; offset /= 2) {
    cost[idx + threadIdx.z] = min(cost[idx + threadIdx.z], __shfl_down_sync(MASK, cost[idx + threadIdx.z], offset, 32));
  }
}

__global__ void compute_disp_kernel(uint8 *disparity,
                                    uint32 height, uint32 width,
                                    uint8 *cost) {
  unsigned int idx_x = blockIdx.x * 4 + threadIdx.x;
  unsigned int idx_y = blockIdx.y * 4 + threadIdx.y;

  unsigned int idx = (idx_y * width + idx_x) * 128;

  if (idx_y < height && idx_x < width) {
    uint8 min_val = UINT8_MAX;
    uint8 best_disp = 0;
    for (int i = 0; i < MAX_DISPARITY; ++i) {
      if (min_val > cost[idx + i]) {
        min_val = cost[idx + i];
        best_disp = i;
      }
    }
    disparity[idx/128] = best_disp;
  }
}

ComputeDisparity::ComputeDisparity(uint8 *disparity,
                                   uint32 height, uint32 width,
                                   uint16 *agg, uint8 *agg0,
                                   uint8 *agg1, uint8 *agg2,
                                   uint8 *agg3, uint8 *agg4,
                                   uint8 *agg5, uint8 *agg6,
                                   uint8 *agg7)
    : height_(height), width_(width),
      pDisparity_(disparity), pAgg0_(agg0),
      pAgg1_(agg1), pAgg2_(agg2),
      pAgg3_(agg3), pAgg4_(agg4),
      pAgg5_(agg5), pAgg6_(agg6),
      pAgg7_(agg7), pAgg_(agg) {}

void ComputeDisparity::inference() {
#if USE_GPU
  compute_disparity_gpu();
#else
  compute_disparity_cpu();
#endif
}

void ComputeDisparity::compute_disparity_gpu() {
  unsigned int grid_dim_x = (width_  + 4 - 1) / 4;
  unsigned int grid_dim_y = (height_ + 4 - 1) / 4;

  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(4, 4);
  compute_disp_kernel<<<grid_dim, block_dim>>>(pDisparity_, height_, width_, pCost_);
}

void ComputeDisparity::compute_disparity_cpu() {
  for (int32 i = 0; i < height_; ++i) {
    for (int32 j = 0; j < width_; ++j) {
      uint16 min_cost = UINT16_MAX;
      uint16 max_cost = 0;
      int32 best_disparity = 0;

      for (int32 d = 0; d < MAX_DISPARITY; ++d) {
        const int32 idx = (i * width_ + j) * MAX_DISPARITY + d;
        const uint16 cost = pAgg0_[idx] + pAgg1_[idx] + pAgg2_[idx] + pAgg3_[idx];
        if (min_cost > cost) {
          min_cost = cost;
          best_disparity = d;
        }
        max_cost = std::max(max_cost, cost);
      }

      if (max_cost != min_cost) {
        pDisparity_[i * width_ + j] = best_disparity;
      } else {
        pDisparity_[i * width_ + j] = UINT16_MAX;
      }
    }
  }

/*  uint8 *cost_ptr = pCost_;

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
  }*/
}