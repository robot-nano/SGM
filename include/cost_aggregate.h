//
// Created by wserver on 2020/7/5.
//

#ifndef SGM_INCLUDE_COST_AGGREGATE_H_
#define SGM_INCLUDE_COST_AGGREGATE_H_

#include "integral_types.h"

class CostAggregate {
 public:
  CostAggregate(uint8 *cost,
                uint8 *cost_agg0, uint8 *cost_agg1,
                uint8 *cost_agg2, uint8 *cost_agg3,
                uint8 *cost_agg4, uint8 *cost_agg5,
                uint8 *cost_agg6, uint8 *cost_agg7,
                int32 p1, int32 p2_init,
                int32 height, int32 width);
  void inference(cv::Mat &img);

  void CostAggregateLeftRight(bool is_forward, uint8 *img, uint8 *pAgg);
  void CostAggregateUpDown(bool is_forward, uint8 *img, uint8 *pAgg);
  void CostAggregateTL2BR(bool is_forward, uint8 *img, uint8 *pAgg);
  void CostAggregateBL2TR(bool is_forward, uint8 *img, uint8 *pAgg);

 private:
  uint8 *pCost_;
  uint8 *pAgg0_, *pAgg1_, *pAgg2_, *pAgg3_;
  uint8 *pAgg4_, *pAgg5_, *pAgg6_, *pAgg7_;
  int32 p1_, p2_init_;
  int32 height_, width_;
};

#endif //SGM_INCLUDE_COST_AGGREGATE_H_
