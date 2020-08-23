/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef SGM_CUDA__COSTS_H_
#define SGM_CUDA__COSTS_H_

#include <cstdint>
#include "config.h"

__global__ void CensusKernelSM2(const uint8_t *im, const uint8_t *im2,
                                cost_t *transform, cost_t *transform2,
                                const uint32_t rows, const uint32_t cols);

#endif //SGM_CUDA__COSTS_H_
