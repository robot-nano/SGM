//
// Created by wserver on 2020/6/21.
//

#ifndef SGM_INCLUDE_SGM_H_
#define SGM_INCLUDE_SGM_H_

#include "integral_types.h"
#include "census_transform.h"
#include "compute_cost.h"
#include "cost_aggregate.h"
#include "compute_disp.h"
#include <memory>

class SGM {
 public:
  SGM(int32 img_height, int32 img_width, int32 window_height, int32 window_width);
  ~SGM();

  void inference(cv::Mat &img_l, cv::Mat &img_r);

 private:
  std::shared_ptr<CensusTransform> pCensusTransform_;
  std::shared_ptr<ComputeCost> pComputeCost_;
  std::shared_ptr<CostAggregate> pComputeCostAgg_;
  std::shared_ptr<ComputeDisparity> pComputeDisparity_;


  uint32 *pCensusL_ = nullptr, *pCensusR_ = nullptr;
  uint8 *pCost_;
  uint8 *pAgg0_;
  uint8 *pDisp_;

  int32 w_hf_h_, w_hf_w_;

#if USE_GPU
  uint8 *pImgL_, *pImgR_;
#endif
};

#endif //SGM_INCLUDE_SGM_H_
