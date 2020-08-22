//
// Created by wserver on 2020/7/23.
//

#ifndef SGM_CUDA__DEBUG_H_
#define SGM_CUDA__DEBUG_H_

#include <iostream>
#include <stdio.h>
#include "config.h"

template <typename T>
void write_file(const char* fname, const T* data, const int size) {
  FILE *fp = fopen(fname, "wb");
  if (fp == NULL) {
    std::cerr << "Couldn't write tranform file" << std::endl;
    exit(-1);
  }
  fwrite(data, sizeof(T), size, fp);
  fclose(fp);
}

void debug_log(const char* str);

#endif //SGM_CUDA__DEBUG_H_
