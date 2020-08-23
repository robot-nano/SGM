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

#ifndef SGM_CUDA__UTIL_H_
#define SGM_CUDA__UTIL_H_

#include <stdio.h>
#include <iostream>

#define GPU_THREADS_PER_BLOCK 256
#define WARP_SIZE 32

#define MASK 0xFFFFFFFF

static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess)
    return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at" << file << ":" << line << std::endl;
  exit(1);
}

__inline__ __device__ int shfl_32(int scalarValue, const int lane) {
  return __shfl_sync(MASK, threadIdx.x, lane);
}

__inline__ __device__ int shfl_up_32(int scalarValue, const int n) {
  return __shfl_up_sync(MASK, scalarValue, n);
}

__inline__ __device__ int shfl_down_32(int scalarValue, const int n) {
  return __shfl_down_sync(MASK, scalarValue, n);
}

__inline__ __device__ int shfl_xor_32(int scalarValue, const int n) {
  return __shfl_xor_sync(MASK, scalarValue, n);
}

__device__ __forceinline__ uint32_t ld_gbl_ca(const __restrict__ uint32_t *addr) {
  uint32_t return_value;
  asm("ld.global.ca.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
  return return_value;
}

__device__ __forceinline__ void st_gbl_cs(const __restrict__ uint32_t *addr, const uint32_t value) {
  asm("st.global.cs.u32 [%0], %1;" :: "l"(addr), "r"(value));
}

__device__ __forceinline__ void uint32_to_uchars(const uint32_t s, int *u1, int *u2, int *u3, int *u4) {
  //*u1 = s & 0xff;
  *u1 = __byte_perm(s, 0, 0x4440);
  //*u2 = (s >> 8) & 0xff;
  *u2 = __byte_perm(s, 0, 0x4441);
  //*u3 = (s >> 16) & 0xff;
  *u3 = __byte_perm(s, 0, 0x4442);
  //*u4 = s >> 24;
  *u4 = __byte_perm(s, 0, 0x4443);
}

__device__ __forceinline__ uint32_t uchars_to_uint32(int u1, int u2, int u3, int u4) {
  return u1 | (u2<<8) | __byte_perm(u3, u4, 0x4077);
}

__device__ __forceinline__ uint32_t uchar_to_uint32(int u1) {
  return __byte_perm(u1, u1, 0x0);
}

template <class T>
__device__ __forceinline__ int popcount(T n) {
#if CSCT or CSCT_PECOMPUTE
  return __popc(n);
#else
  return __popcll(n);
#endif
}

__inline__ __device__ int warpReduceMinIndex2(int *val, int idx) {
  for (int d = 1; d < MAX_DISPARITY; d *= 2) {
    int tmp = shfl_xor_32(*val, d);
    int tmp_idx = shfl_xor_32(idx, d);
    if (*val > tmp) {
      *val = tmp;
      idx = tmp_idx;
    }
  }
  return idx;
}

__inline__ __device__ int warpReduceMinIndex(int val, int idx) {
  for (int d = 1; d < WARP_SIZE; d *= 2) {
    int tmp = shfl_xor_32(val, d);
    int tmp_idx = shfl_xor_32(idx, d);
    if (val > tmp) {
      val = tmp;
      idx = tmp_idx;
    }
  }
  return idx;
}

__inline__ __device__ int warpReduceMin(int val) {
  val = min(val, shfl_xor_32(val, 1));
  val = min(val, shfl_xor_32(val, 2));
  val = min(val, shfl_xor_32(val, 4));
  val = min(val, shfl_xor_32(val, 8));
  val = min(val, shfl_xor_32(val, 16));
  return val;
}

#endif //SGM_CUDA__UTIL_H_
