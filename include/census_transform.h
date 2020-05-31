//
// Created by wserver on 2020/5/31.
//

#ifndef SGM_INCLUDE_CENSUS_TRANSFORM_H_
#define SGM_INCLUDE_CENSUS_TRANSFORM_H_

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include "integral_types.h"
#include "config.h"

class CensusTransform {
 public:
  CensusTransform(int window_height, int window_width);

  void inference(cv::Mat &img_left, cv::Mat &img_right,
                 uint8 *l_result, uint8 *r_result);

 private:
  /**
   * use cpu calculate census_transform
   * @param img_left    left img store im cpu host with opencv format
   * @param transform_left  transformed result
   */
  void census_transform_cpu(cv::Mat &img,
                            uint8 *t_result);
  __global__ void census_transform_gpu();

  int window_height_;
  int window_width_;
};

#endif //SGM_INCLUDE_CENSUS_TRANSFORM_H_
