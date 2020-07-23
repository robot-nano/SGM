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
  CensusTransform(int32 img_height, int32 img_width,
                  int window_height, int window_width,
                  uint32 *census_l, uint32 *census_r);
  ~CensusTransform();

  void census_inference(void *img_left, void *img_right, uint32 *census_l, uint32 *census_r);

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
  void census_transform_gpu(const uint8 *im, const uint8 *im2,
                            uint32 *transform, uint32 *transform2,
                            const uint32 rows, const uint32 cols);

  int w_hf_h_;  // half of window size, window size is odd
  int w_hf_w_;

  uint32 *pCensusL_, *pCensusR_;
  uint32 imgHeight_, imgWidth_;
};

#endif //SGM_INCLUDE_CENSUS_TRANSFORM_H_
