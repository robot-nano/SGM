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
