#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>

using namespace at;
Tensor conv_depthwise3d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    const std::array<bool, 3> output_mask);

Tensor conv_depthwise3d_cuda(
    const Tensor& input,
    const Tensor& weight,
    int kernel_size_one,
    const Tensor& bias,
    int stride_one,
    int padding_one,
    int dilation_one);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_depthwise3d_backward_cuda", &conv_depthwise3d_backward_cuda,
        "conv_depthwise3d_backward_cuda");
  m.def("conv_depthwise3d_cuda", &conv_depthwise3d_cuda,
        "conv_depthwise3d_cuda");
}
