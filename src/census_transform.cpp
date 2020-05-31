//
// Created by wserver on 2020/5/31.
//

#include "census_transform.h"

CensusTransform::CensusTransform(int window_height, int window_width)
  : window_height_(window_height), window_width_(window_width)
{}

void CensusTransform::inference(cv::Mat &img_left, cv::Mat &img_right,
                                uint8 *l_result, uint8 *r_result) {
  if (USE_GPU) {

  } else {
    census_transform_cpu(img_left, l_result);
    census_transform_cpu(img_right, r_result);
  }
}

void CensusTransform::census_transform_cpu(cv::Mat &img,
                                           uint8 *t_result) {
  int w_hf_h = window_height_ / 2;    // half of window height
  int w_hf_w = window_width_ / 2;     // half of window width
  int img_height  = img.rows;
  int img_width   = img.cols;

  for (int32 i = w_hf_h; i < img_height - w_hf_h; ++i) {
    for (int32 j = w_hf_w; j < img_width - w_hf_w; ++j) {
      // central of window
      const uint8 gray_center = img.at<uchar>(j, i);

      uint32 census_val = 0u;
      for (int32 wh = -w_hf_h; wh <= w_hf_h; ++wh) {
        for (int32 ww = -w_hf_w; ww <= w_hf_w; ++ww) {
          census_val <<= 1;
          const uint8 gray = img.at<uchar>(i + wh, j + ww);
          if (gray < gray_center)
            census_val += 1;
        }
      }

      t_result[i * img_width + j] = census_val;
    }
  }
}