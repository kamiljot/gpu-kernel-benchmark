/**
 * @file    add_kernel_cuda.cu
 * @brief   CUDA global memory variant for add kernel (kernel launcher).
 * @author  Kamil J.
 * @date    2025-07-10
 */

#include <cuda_runtime.h>
#include <cstddef>

__global__
void add_global_kernel(const float* in1, const float* in2, float* out, size_t size) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) out[idx] = in1[idx] + in2[idx];
}

// Kernel launcher for backend dispatch
extern "C"
void launch_cuda_add_global(float* in1, float* in2, float* out, size_t size) {
	int threads = 256;
	int blocks = (size + threads - 1) / threads;
	add_global_kernel << <blocks, threads >> > (in1, in2, out, size);
	cudaDeviceSynchronize();
}