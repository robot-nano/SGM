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

#ifndef SGM_CUDA__HAMMING_COST_H_
#define SGM_CUDA__HAMMING_COST_H_

#include "config.h"
#include "util.h"
#include <cstdint>

__global__ void
HammingDistanceCostKernel(const cost_t *d_transform0, const cost_t *d_transform1,
                          uint8_t *d_cost, const int rows, const int cols);

#endif //SGM_CUDA__HAMMING_COST_H_
