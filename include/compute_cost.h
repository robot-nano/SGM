//
// Created by wserver on 2020/6/9.
//

#ifndef SGM_INCLUDE_COMPUTE_COST_H_
#define SGM_INCLUDE_COMPUTE_COST_H_

#include "util.h"
#include "integral_types.h"
#include "config.h"


class ComputeCost {
 public:
  ComputeCost(uint8 *cost,
      uint32 *census_left, uint32 *census_right,
      int32 max_disparity,
      uint32 height, uint32 width);
  void inference();

 private:
  void compute_cost_cpu();

  uint16 Hamming32(const uint32& x, const uint32& y);

  int32 max_disparity_;
  uint32 *p_census_l_, *p_census_r_;
  uint8 *p_cost_;
  uint32 height_, width_;
};

#endif //SGM_INCLUDE_COMPUTE_COST_H_
