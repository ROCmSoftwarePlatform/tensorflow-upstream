/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#if TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/gpu_fusion_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "hip/hip_fp16.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace rocm_kernels {

//-------------------------------------------------------------------
__global__ void AddReluKernel(int nthreads, const float* in0, const float* in1,
                              float* out) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    out[index] = fmaxf(0.0f, in0[index] + in1[index]);
  }
}

void FusionAddRelu(OpKernelContext* ctx, const float* in0, const float* in1,
                   float* out, unsigned N) {
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(N, d);
  GPU_LAUNCH_KERNEL(AddReluKernel, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, in0, in1, out);
}

__global__ void AddReluKernel(int nthreads, const half* in0, const half* in1,
                              half* out) {
  half hZero(0.0f16);

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    half sum = __hadd(in0[index], in1[index]);
    out[index] = __hgt(sum, hZero) ? sum : hZero;
  }
}

void FusionAddRelu(OpKernelContext* ctx, const Eigen::half* in0,
                   const Eigen::half* in1, Eigen::half* out, unsigned N) {
  const half* h_in0 = reinterpret_cast<const half*>(in0);
  const half* h_in1 = reinterpret_cast<const half*>(in1);
  half* h_out = reinterpret_cast<half*>(out);

  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(N, d);
  GPU_LAUNCH_KERNEL(AddReluKernel, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, h_in0, h_in1, h_out);
}
//-------------------------------------------------------------------
__global__ void AddNReluGradKernel(int nthreads, const float* in0,
                                   const float* in1, const float* in2,
                                   float* out) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    out[index] = (in2[index] > 0.0f) ? (in0[index] + in1[index]) : 0.0f;
  }
}

void FusionAddNReluGrad(OpKernelContext* ctx, const float* in0,
                        const float* in1, const float* in2, float* out,
                        unsigned N) {
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(N, d);
  GPU_LAUNCH_KERNEL(AddNReluGradKernel, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, in0, in1, in2, out);
}

__global__ void AddNReluGradKernel(int nthreads, const half* in0,
                                   const half* in1, const half* in2,
                                   half* out) {
  half hZero(0.0f16);

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    out[index] =
        __hgt(in2[index], hZero) ? __hadd(in0[index], in1[index]) : hZero;
  }
}

void FusionAddNReluGrad(OpKernelContext* ctx, const Eigen::half* in0,
                        const Eigen::half* in1, const Eigen::half* in2,
                        Eigen::half* out, unsigned N) {
  const half* h_in0 = reinterpret_cast<const half*>(in0);
  const half* h_in1 = reinterpret_cast<const half*>(in1);
  const half* h_in2 = reinterpret_cast<const half*>(in2);
  half* h_out = reinterpret_cast<half*>(out);

  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(N, d);
  GPU_LAUNCH_KERNEL(AddNReluGradKernel, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, h_in0, h_in1, h_in2, h_out);
}
//-------------------------------------------------------------------

}  // namespace rocm_kernels

}  // namespace tensorflow

#endif
