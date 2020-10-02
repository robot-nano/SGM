//
// Created by wserver on 2020/8/29.
//

#include <torch/torch.h>
#include <iostream>

at::Tensor BatchNorm_Forward_CUDA(
    const at::Tensor input_,
    const at::Tensor ex_,
    const at::Tensor exs_,
    const at::Tensor gamma_,
    const at::Tensor beta_,
    float eps);



int main(int argc, char **argv) {

  auto input = torch::rand({2, 3, 3});
  auto ex = torch::ones({1});
  auto exs = torch::ones({1});
  auto gamma = torch::ones({1});
  auto beta = torch::ones({1});

  auto out = BatchNorm_Forward_CUDA(input, ex, exs, gamma, beta, 0.2f);
}