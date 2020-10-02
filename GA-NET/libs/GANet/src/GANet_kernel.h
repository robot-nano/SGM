#include <torch/extension.h>
#include "c_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

GANET_DLL void sga_kernel_forward(at::Tensor input, at::Tensor guidance_down,
                                  at::Tensor guidance_up, at::Tensor guidance_right,
                                  at::Tensor guidance_left, at::Tensor temp_out,
                                  at::Tensor output, at::Tensor mask);

GANET_DLL void sga_kernel_backward(at::Tensor input, at::Tensor guidance_down,
                                   at::Tensor guidance_up, at::Tensor guidance_right,
                                   at::Tensor guidance_left, at::Tensor temp_out,
                                   at::Tensor mask, at::Tensor max_idx,
                                   at::Tensor gradOutput, at::Tensor temp_grad,
                                   at::Tensor gradInput, at::Tensor grad_down,
                                   at::Tensor grad_up, at::Tensor grad_right,
                                   at::Tensor grad_left);

GANET_DLL void lga_backward(at::Tensor input, at::Tensor filters,
                            at::Tensor gradOutput, at::Tensor gradInput,
                            at::Tensor gradFilters, const int radius);

GANET_DLL void lga_forward(at::Tensor input, at::Tensor filters, at::Tensor output,
                           const int radius);

GANET_DLL void lga3_backward(at::Tensor input, at::Tensor filters,
                             at::Tensor gradOutput, at::Tensor gradInput,
                             at::Tensor gradFilters, const int radius);

GANET_DLL void lga3_forward(at::Tensor input, at::Tensor filters, at::Tensor output,
                            const int radius);

#ifdef __cplusplus
}
#endif