//
// Created by wserver on 2020/7/25.
//

#ifndef SGM_CUDA__HAMMING_COST_H_
#define SGM_CUDA__HAMMING_COST_H_

#include "config.h"
#include "util.h"
#include <cstdint>

__global__ void
HammingDistanceCostKernel(const cost_t *d_transform0, const cost_t *d_transform1,
                          uint8_t *d_cost, const int rows, const int cols);

#endif //SGM_CUDA__HAMMING_COST_H_
