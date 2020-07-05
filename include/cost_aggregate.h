//
// Created by wserver on 2020/7/5.
//

#ifndef SGM_INCLUDE_COST_AGGREGATE_H_
#define SGM_INCLUDE_COST_AGGREGATE_H_

#include "integral_types.h"

class CostAggregate {
 public:
  CostAggregate(uint8 *cost, uint8 *pCostAgg0,
                int32 p1, int32 p2_init,
                int32 height, int32 width);
  void inference(cv::Mat &img);

  void CostAggregateLeftRight(bool is_forward, uint8 *img);

 private:
  uint8 *pCost_;
  uint8 *pCostAgg0_;
  int32 p1_, p2_init_;
  int32 height_, width_;
};

#endif //SGM_INCLUDE_COST_AGGREGATE_H_
