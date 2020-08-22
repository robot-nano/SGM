//
// Created by wserver on 2020/7/23.
//

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
