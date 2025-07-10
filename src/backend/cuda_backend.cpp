/**
 * @file    cuda_backend.cpp
 * @brief   Implements CudaBackend class (simple CUDA interface).
 * @author  Kamil J.
 * @date    2025-07-10
 */

#include "backend/cuda_backend.hpp"
#include "kernels/kernel_registry.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "kernels/add/add_kernel_cuda.hpp"


std::string CudaBackend::name() const {
	return "cuda";
}

float* CudaBackend::allocate(size_t num_elements) {
	float* dev_ptr = nullptr;
	cudaError_t err = cudaMalloc(&dev_ptr, num_elements * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "[CUDA] cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
		return nullptr;
	}
	return dev_ptr;
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
	}
	else {
		std::cerr << "[CUDA] Kernel not implemented: " << kernel_name << std::endl;
		assert(false && "Unknown CUDA kernel");
	}
}