/**
 * @file    cuda_backend.cpp
 * @brief   CUDA backend implementation for gpu-kernel-benchmark.
 * @author  Kamil J.
 * @date    2025-07-10
 */

#ifdef USE_CUDA

#include "backend/cuda_backend.hpp"
#include <cuda_runtime.h>
#include <string>

 // Kernel launcher (from add_kernel_cuda.cu)
extern "C" void launch_cuda_add_global(float*, float*, float*, size_t);

std::string CudaBackend::name() const { return "cuda"; }

float* CudaBackend::allocate(size_t num_elements) {
	float* ptr = nullptr;
	cudaMalloc(&ptr, num_elements * sizeof(float));
	return ptr;
}

void CudaBackend::free(float* ptr) {
	cudaFree(ptr);
}

void CudaBackend::copy_to_device(float* dst, const float* src, size_t num_elements) {
	cudaMemcpy(dst, src, num_elements * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaBackend::copy_to_host(float* dst, const float* src, size_t num_elements) {
	cudaMemcpy(dst, src, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
}

void CudaBackend::launch_kernel(const std::string& kernel_name,
	float* input1, float* input2, float* output, size_t size) {
	if (kernel_name == "add_global") {
		launch_cuda_add_global(input1, input2, output, size);
		cudaDeviceSynchronize();
	}
	// TODO: add more kernels here
}

#endif // USE_CUDA