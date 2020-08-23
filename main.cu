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

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "disparity.h"

int main(int argc, char** argv) {
  uint8_t p1, p2;
  p1 = atoi(argv[1]);
  p2 = atoi(argv[2]);

  init_disparity_method(p1, p2);

  cv::Mat im = cv::imread(argv[3]);
  if (!im.data) {
    std::cerr << "Couldn't read the file " << argv[3] << std::endl;
    exit(EXIT_FAILURE);
  }
  cv::Mat im2 = cv::imread(argv[4]);
  if (!im2.data) {
    std::cerr << "Couldn't read the file " << argv[4] << std::endl;
    exit(EXIT_FAILURE);
  }

  if (im.channels() > 1) {
    cv::cvtColor(im, im, CV_RGB2GRAY);
  }
  if (im2.channels() > 1) {
    cv::cvtColor(im2, im2, CV_RGB2GRAY);
  }

  if (im.rows % 4 || im.cols % 4) {
    cv::resize(im, im, cv::Size(im.cols/4*4, im.rows/4*4));
    cv::resize(im2, im2, cv::Size(im.cols/4*4, im.rows/4*4));
  }

  compute_disparity_method(im, im2);
}
