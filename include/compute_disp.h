//
// Created by wserver on 2020/6/11.
//

#ifndef SGM_INCLUDE_COMPUTE_DISP_H_
#define SGM_INCLUDE_COMPUTE_DISP_H_

#include "config.h"
#include "integral_types.h"

class ComputeDisparity {
 public:
  ComputeDisparity(uint8 *disparity, uint32 height, uint32 width,
                   uint16 *agg, uint8 *agg0,
                   uint8 *agg1, uint8 *agg2,
                   uint8 *agg3, uint8 *agg4,
                   uint8 *agg5, uint8 *agg6,
                   uint8 *agg7);
  void inference();
 private:
  void compute_disparity_gpu();
  void compute_disparity_cpu();
  uint32 height_, width_;
  uint8 *pDisparity_;
  uint8 *pAgg0_, *pAgg1_, *pAgg2_, *pAgg3_, *pAgg4_, *pAgg5_, *pAgg6_, *pAgg7_;
  uint16 *pAgg_;
  uint8 *pCost_;
};

#endif //SGM_INCLUDE_COMPUTE_DISP_H_
