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

#include "hamming_cost.h"

__global__ void
HammingDistanceCostKernel(const cost_t *d_transform0, const cost_t *d_transform1,
                          uint8_t *d_cost, const int rows, const int cols) {
  const int y = blockIdx.x;
  const int THRid = threadIdx.x;

  __shared__ cost_t SharedMatch[2 * MAX_DISPARITY];
  __shared__ cost_t SharedBase[MAX_DISPARITY];

  SharedMatch[MAX_DISPARITY + THRid] = d_transform1[y * cols];

  int n_iter = cols / MAX_DISPARITY;
  for (int ix = 0; ix < n_iter; ++ix) {
    const int x = ix * MAX_DISPARITY;
    SharedMatch[THRid] = SharedMatch[THRid + MAX_DISPARITY];
    SharedMatch[THRid + MAX_DISPARITY] = d_transform1[y * cols + x + THRid];
    SharedBase[THRid] = d_transform0[y * cols + x + THRid];

    __syncthreads();
    for (int i = 0; i < MAX_DISPARITY; ++i) {
      const cost_t base = SharedBase[i];
      const cost_t match = SharedMatch[MAX_DISPARITY + i - THRid];
      d_cost[(y * cols + x + i) * MAX_DISPARITY + THRid] = popcount(base ^ match);
    }
    __syncthreads();
  }

  const int x = MAX_DISPARITY * (cols / MAX_DISPARITY);
  const int left = cols - x;
  if (left > 0) {
    SharedMatch[THRid] = SharedMatch[THRid + MAX_DISPARITY];
    if (THRid < left) {
      SharedMatch[THRid + MAX_DISPARITY] = d_transform1[y * cols + x + THRid];
      SharedBase[THRid] = d_transform0[y * cols + x + THRid];
    }

    __syncthreads();
    for (int i = 0; i < left; ++i) {
      const cost_t base = SharedBase[i];
      const cost_t match = SharedMatch[MAX_DISPARITY + i - THRid];
      d_cost[(y * cols + x + i) * MAX_DISPARITY + THRid] = popcount(base ^ match);
    }
    __syncthreads();
  }
}