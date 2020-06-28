//
// Created by wserver on 2020/6/11.
//

#ifndef SGM_INCLUDE_COMPUTE_DISP_H_
#define SGM_INCLUDE_COMPUTE_DISP_H_

#include "config.h"
#include "integral_types.h"

class ComputeDisparity {
 public:
  ComputeDisparity(uint8 *disparity, uint32 height, uint32 width, uint8 *cost);
  void inference();
 private:
  void compute_disparity_gpu();
  void compute_disparity_cpu();
  uint32 height_, width_;
  uint8 *pDisparity_;
  uint8 *pCost_;
};

#endif //SGM_INCLUDE_COMPUTE_DISP_H_
