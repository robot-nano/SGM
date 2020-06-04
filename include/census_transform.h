//
// Created by wserver on 2020/5/31.
//

#ifndef SGM_INCLUDE_CENSUS_TRANSFORM_H_
#define SGM_INCLUDE_CENSUS_TRANSFORM_H_
#include "config.h"

#if USE_GPU
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#endif
#include <opencv2/core/core.hpp>
#include "integral_types.h"
#include "config.h"

class CensusTransform {
 public:
  CensusTransform(int window_height, int window_width);

  void inference(uint32 *l_result, uint32 *r_result,
                 void *img_left, void *img_right,
                 int img_rows = 0, int img_cols = 0);

 private:
  /**
   * use cpu calculate census_transform
   * @param img_left    left img store im cpu host with opencv format
   * @param transform_left  transformed result
   */
  void census_transform_cpu(cv::Mat &img,
                            uint32 *t_result);

  // blockDim.x = blockDim.y = 16
  // | block1        block2      ... blockn|
  // | block(n+1)    block(n+2) ... block2n|
  // |              ...                    |
  void census_transform_gpu(uint8 *img, uint32 *result,
                            int32 img_rows, int32 img_cols);


  int window_height_;
  int window_width_;
  int w_hf_h_;  // half of window size, window size is odd
  int w_hf_w_;
};

#endif //SGM_INCLUDE_CENSUS_TRANSFORM_H_
