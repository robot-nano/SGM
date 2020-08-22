//
// Created by wserver on 2020/7/24.
//

#ifndef SGM_CUDA__COSTS_H_
#define SGM_CUDA__COSTS_H_

#include <cstdint>
#include "config.h"

__global__ void CensusKernelSM2(const uint8_t *im, const uint8_t *im2,
                                cost_t *transform, cost_t *transform2,
                                const uint32_t rows, const uint32_t cols);

#endif //SGM_CUDA__COSTS_H_
