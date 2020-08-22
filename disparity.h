//
// Created by wserver on 2020/7/23.
//

#ifndef SGM_CUDA__DISPARITY_H_
#define SGM_CUDA__DISPARITY_H_

#include <opencv2/core/core.hpp>
#include "config.h"
#include "util.h"
#include "debug.h"
#include "costs.h"
#include "hamming_cost.h"
#include "cost_aggregation.h"

void init_disparity_method(const uint8_t p1, const uint8_t p2);
cv::Mat compute_disparity_method(cv::Mat &left, cv::Mat &right);
void finish_disparity_method();
static void free_memory();

#endif //SGM_CUDA__DISPARITY_H_
