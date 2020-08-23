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
