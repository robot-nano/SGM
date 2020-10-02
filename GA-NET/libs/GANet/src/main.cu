//
// Created by wserver on 2020/8/19.
//

#include "GANet_kernel.h"
#include <torch/torch.h>

int main(int argc, char **argv) {
  torch::Tensor tensor = torch::rand({1, 3, 5, 10, 10}).cuda();
  torch::Tensor guidance_down = tensor;
  torch::Tensor guidance_up = tensor;
  torch::Tensor guidance_left = tensor;
  torch::Tensor guidance_right = tensor;
  torch::Tensor temp_out = tensor;
  torch::Tensor output = tensor;
  torch::Tensor mask = tensor;
//  torch::Tensor guidance_down, guidance_up, guidance_right, guidance_left, temp_out, output, mask;
  sga_kernel_forward(tensor, guidance_down, guidance_up, guidance_right, guidance_left,
                     temp_out, output, mask);
  float *out_ptr = temp_out.data<float>();
  std::cout << " ";
}