/**
 * @file    cpu_backend.cpp
 * @brief   Implements CpuBackend class.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Simple backend for CPU compute and memory operations.
 */

#include "backend/cpu_backend.hpp"
#include "kernels/kernel_registry.hpp"
#include <cstring>

std::string CpuBackend::name() const {
	return "cpu";
}

float* CpuBackend::allocate(size_t num_elements) {
	return new (std::nothrow) float[num_elements];
}

void CpuBackend::free(float* ptr) {
	delete[] ptr;
}

void CpuBackend::copy_to_device(float* dst, const float* src, size_t num_elements) {
	std::memcpy(dst, src, num_elements * sizeof(float));
}

void CpuBackend::copy_to_host(float* dst, const float* src, size_t num_elements) {
	std::memcpy(dst, src, num_elements * sizeof(float));
}

void CpuBackend::launch_kernel(const std::string& kernel_name,
	float* input1, float* input2, float* output, size_t size) {
	auto kernel = KernelRegistry::instance().create(kernel_name);
	if (kernel)
		kernel->run(input1, input2, output, size);
	// else: could throw, log error, etc.
}