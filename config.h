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

#ifndef SGM_CUDA__CONFIG_H_
#define SGM_CUDA__CONFIG_H_

#include <cstdint>

#define LOG           false
#define WRITE_FILES   true

#define PATH_AGGREGATION  8
#define MAX_DISPARITY     128
#define CENSUS_WIDTH      9
#define CENSUS_HEIGHT     7

#define TOP               (CENSUS_HEIGHT - 1) / 2
#define LEFT              (CENSUS_WIDTH - 1) / 2

typedef uint32_t cost_t;
#define MAX_COST          30

#define BLOCK_SIZE        256
#define COSTAGG_BLOCKSIZE         GPU_THREADS_PER_BLOCK
#define COSTAGG_BLOCKSIZE_HORIZ   GPU_THREADS_PER_BLOCK

#endif //SGM_CUDA__CONFIG_H_
