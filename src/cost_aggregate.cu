//
// Created by wserver on 2020/7/5.
//

#include <algorithm>
#include <opencv2/core/core.hpp>
#include "cost_aggregate.h"
#include "config.h"

CostAggregate::CostAggregate(uint8 *cost,
                             uint8 *cost_agg0, uint8 *cost_agg1,
                             uint8 *cost_agg2, uint8 *cost_agg3,
                             uint8 *cost_agg4, uint8 *cost_agg5,
                             uint8 *cost_agg6, uint8 *cost_agg7,
                             int32 p1, int32 p2_init,
                             int32 height, int32 width)
    : pCost_(cost), pAgg0_(cost_agg0), pAgg1_(cost_agg1),
      pAgg2_(cost_agg2), pAgg3_(cost_agg3), pAgg4_(cost_agg4),
      pAgg5_(cost_agg5), pAgg6_(cost_agg6), pAgg7_(cost_agg7),
      p1_(p1), p2_init_(p2_init),
      height_(height), width_(width) {

}

void CostAggregate::inference(cv::Mat &img) {
  CostAggregateLeftRight(true, img.data, pAgg0_);
  CostAggregateLeftRight(false, img.data, pAgg1_);

  CostAggregateUpDown(true, img.data, pAgg2_);
  CostAggregateUpDown(false, img.data, pAgg3_);
}

void CostAggregate::CostAggregateLeftRight(bool is_forward, uint8 *img, uint8 *agg) {
  const int32 direction = is_forward ? 1 : -1;
  for (int32 h = 0; h < height_; ++h) {
    uint8 *cost_init_idx = (is_forward) ? (pCost_ + h * width_ * MAX_DISPARITY)
                                        : (pCost_ + ((h + 1) * width_ - 1) * MAX_DISPARITY);
    uint8 *cost_agg_idx = (is_forward) ? (agg + h * width_ * MAX_DISPARITY)
                                       : (agg + ((h + 1) * width_ - 1) * MAX_DISPARITY);
    uint8 *img_idx = (is_forward) ? (img + (h + 2) * (width_ + 4))
                                  : (img + (h + 2 + 1) * (width_ + 4) - 1);

    uint8 gray = *img_idx;
    uint8 gray_last = *img_idx;

    memcpy(cost_agg_idx, cost_init_idx, MAX_DISPARITY * sizeof(uint8));
    cost_init_idx += direction * MAX_DISPARITY;
    img_idx += direction;

    uint8 min_cost_last_path = *std::min_element(cost_agg_idx, cost_agg_idx + MAX_DISPARITY);

    for (int32 w = 1; w < width_; ++w) {
      gray = *img_idx;
      uint8 min_cost = UINT8_MAX;
      for (int32 d = 0; d < MAX_DISPARITY; ++d) {
        const uint8 cost = cost_init_idx[d];
        const uint16 l1 = cost_agg_idx[d];
        uint16 l2 = UINT16_MAX;
        if (d != 0)
          l2 = cost_agg_idx[d - 1] + p1_;
        uint16 l3 = UINT16_MAX;
        if (d != MAX_DISPARITY - 1)
          l3 = cost_agg_idx[d + 1] + p1_;
        const uint16 l4 = min_cost_last_path + std::max(p1_, p2_init_ / (abs(gray - gray_last) + 1));

        const uint8 cost_s =
            cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - min_cost_last_path);

        *(cost_agg_idx + d + direction * MAX_DISPARITY) = cost_s;
        min_cost = std::min(min_cost, cost_s);
      }

      min_cost_last_path = min_cost;

      cost_init_idx += direction * MAX_DISPARITY;
      cost_agg_idx += direction * MAX_DISPARITY;
      img_idx += direction;

      gray_last = gray;
    }
  }
}

void CostAggregate::CostAggregateUpDown(bool is_forward, uint8 *img, uint8 *agg) {
  const int32 direction = is_forward ? 1 : -1;
  for (int32 w = 0; w < width_; ++w) {
    uint8 *cost_init_idx = (is_forward) ? (pCost_ + w * MAX_DISPARITY)
                                        : (pCost_ + ((height_ - 1) * width_ + w) * MAX_DISPARITY);
    uint8 *cost_agg_idx  = (is_forward) ? (agg + w * MAX_DISPARITY)
                                        : (agg + ((height_ - 1) * width_ + w) * MAX_DISPARITY);
    uint8 *img_idx = (is_forward) ? (img + (width_ + 4) * 2 +  w + 2)
                                  : (img + (height_ + 1) * (width_ + 4) + w + 2);

    uint8 gray = *img_idx;
    uint8 gray_last = *img_idx;

    memcpy(cost_agg_idx, cost_init_idx, MAX_DISPARITY * sizeof(uint8));
    cost_init_idx += direction * width_ * MAX_DISPARITY;
    img_idx += direction * (width_ + 4);

    uint8 min_cost_last_path = *std::min_element(cost_agg_idx, cost_agg_idx + MAX_DISPARITY);

    for (int32 h = 1; h < height_; ++h) {
      gray = *img_idx;
      uint8 min_cost = UINT8_MAX;
      for (int32 d = 0; d < MAX_DISPARITY; ++d) {
        const uint8 cost = cost_init_idx[d];
        const uint16 l1 = cost_agg_idx[d];
        uint16 l2 = UINT16_MAX;
        if (d != 0)
          l2 = cost_agg_idx[d - 1] + p1_;
        uint16 l3 = UINT16_MAX;
        if (d != MAX_DISPARITY - 1)
          l3 = cost_agg_idx[d + 1] + p1_;
        const uint16 l4 = min_cost_last_path + std::min(p1_, p2_init_ / (abs(gray - gray_last) + 1));

        const uint8 cost_s =
            cost + static_cast<uint8>(std::min(std::min(l1, l2), std::min(l3, l4)) - min_cost_last_path);

        *(cost_agg_idx + d + direction * width_ * MAX_DISPARITY) = cost_s;
        min_cost = std::min(min_cost, cost_s);
      }

      min_cost_last_path = min_cost;

      cost_init_idx += direction * width_ * MAX_DISPARITY;
      cost_agg_idx += direction * width_ * MAX_DISPARITY;
      img_idx += direction * (width_ + 4);

      gray_last = gray;
    }
  }
}