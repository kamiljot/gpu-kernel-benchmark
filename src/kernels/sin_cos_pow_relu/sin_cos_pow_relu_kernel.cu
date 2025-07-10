/**
 * @file    sin_cos_pow_relu_kernel.cu
 * @brief   CUDA kernel and implementation for SinCosPowReluKernel class.
 * @author  Kamil J.
 * @date    2025-07-09
 */

#include <cuda_runtime.h>

#include <cmath>

#include "sin_cos_pow_relu_kernel.hpp"

 /**
  * @brief CUDA kernel for sin, cos, pow, and ReLU operation.
  */
__global__ void sin_cos_pow_relu_kernel_kernel(const float* a, const float* b, float* out, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		float tmp = sinf(a[idx]) + cosf(b[idx]);
		tmp = powf(tmp, 2.0f);
		out[idx] = tmp > 0.0f ? tmp : 0.0f;  // ReLU
	}
}

void SinCosPowReluKernel::launch(const std::vector<float>& input_a, const std::vector<float>& input_b,
	std::vector<float>& output)
{
	int N = static_cast<int>(input_a.size());
	float* d_a, * d_b, * d_out;
	cudaMalloc(&d_a, N * sizeof(float));
	cudaMalloc(&d_b, N * sizeof(float));
	cudaMalloc(&d_out, N * sizeof(float));

	cudaMemcpy(d_a, input_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, input_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = (N + blockSize - 1) / blockSize;
	sin_cos_pow_relu_kernel_kernel << <gridSize, blockSize >> > (d_a, d_b, d_out, N);

	cudaMemcpy(output.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
}