//
// Created by wserver on 2020/6/26.
//

#include <cuda.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>

#define MASK 0xFFFFFFFF

__global__ void kernel(int *a) {
  for (int i = 1; i < 4; ++i) {
    a[threadIdx.x] = min(a[threadIdx.x], a[threadIdx.x + i * threadIdx.x]);
  }
  for (int offset = 16; offset >= 1; offset /= 2) {
    a[threadIdx.x] = min(a[threadIdx.x], __shfl_down_sync(MASK, a[threadIdx.x], offset, 32));
  }
}

__global__ void kernel2(int *a) {
  if (threadIdx.x < 10) {
    printf("%d \n", (10 - threadIdx.x) > 0);
    a[threadIdx.x] = -1;
  }
}

int main() {
  uint32_t *data = new uint32_t[128];
  for (int i = 0; i < 128; ++i) {
    data[i] = 128 - i;
  }

  std::cout << *std::min_element(data, data + 128) << std::endl;


//  int a[128];
//  for (int i = 0; i < 128; ++i) {
//    a[i] = rand() % 1000;
//  }
//  int min = INT_MAX;
//  for (int i = 0; i < 128; ++i) {
//    if (min > a[i])
//      min = a[i];
//  }
//  std::cout << min << std::endl;
//
//  int *dev_a;
//  cudaMalloc(&dev_a, 128 * sizeof(int));
//  cudaMemcpy(dev_a, a, 128 * sizeof(int), cudaMemcpyHostToDevice);
//
//  kernel2<<<1, 128>>>(dev_a);
//  int result[128];
//  cudaMemcpy(result, dev_a, 128 * sizeof(int), cudaMemcpyDeviceToHost);

//  std::cout << result[0] << std::endl;
//  for (int i = 0; i < 128; ++i) {
//    std::cout << result[i] << std::endl;
//  }
}
