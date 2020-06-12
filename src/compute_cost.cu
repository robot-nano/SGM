//
// Created by wserver on 2020/6/9.
//

#include "compute_cost.h"

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