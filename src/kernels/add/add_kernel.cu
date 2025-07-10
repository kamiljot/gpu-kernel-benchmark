/**
 * @file    add_kernel.cu
 * @brief   CUDA kernel and implementation for AddKernel class.
 * @author  Kamil J.
 * @date    2025-07-09
 */

#include <cuda_runtime.h>
#include "add_kernel.hpp"

 /**
  * @brief CUDA kernel for element-wise addition.
  */
__global__ void add_kernel_kernel(const float* a, const float* b, float* out, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		out[idx] = a[idx] + b[idx];
	}
}

void AddKernel::launch(const std::vector<float>& input_a, const std::vector<float>& input_b,
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
	add_kernel_kernel << <gridSize, blockSize >> > (d_a, d_b, d_out, N);

	cudaMemcpy(output.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);
}
