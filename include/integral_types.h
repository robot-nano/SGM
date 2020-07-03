//
// Created by wserver on 2020/5/31.
//

#ifndef SGM_INCLUDE_INTEGRAL_TYPES_H_
#define SGM_INCLUDE_INTEGRAL_TYPES_H_

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdio>

typedef uint64_t  uint64;
typedef int64_t   int64;
typedef uint32_t  uint32;
typedef int32_t   int32;
typedef uint16_t  uint16;
typedef int16_t   int16;
typedef uint8_t   uint8;
typedef int8_t    int8;
typedef float     float32;
typedef double    float64;

#define MASK 0xFFFFFFFF

#define CudaSafeCall(error) cuda_safe_call(error, __FILE__, __LINE__)

inline void cuda_safe_call(cudaError error, const char* file, const int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "cuda error %s : %d %s\n", file, line, cudaGetErrorString(error));
    exit(-1);
  }
}

#endif //SGM_INCLUDE_INTEGRAL_TYPES_H_
