//
// Created by wserver on 2020/6/11.
//

#include "compute_disp.h"
#include <algorithm>

ComputeDisparity::ComputeDisparity(uint8 *disparity,
                                   uint32 height, uint32 width,
                                   uint8 *cost)
    : height_(height), width_(width),
      pDisparity_(disparity), pCost_(cost)
{}

void ComputeDisparity::inference() {
#if USE_GPU
#else
  compute_disparity_cpu();
#endif
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