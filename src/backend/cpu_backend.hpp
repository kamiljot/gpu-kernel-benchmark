/**
 * @file    cpu_backend.hpp
 * @brief   CPU implementation of BackendInterface.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Implements a backend for CPU operations.
 */

#pragma once

#include "../include/backend/backend_interface.hpp"

 /**
  * @class   CpuBackend
  * @brief   CPU implementation of BackendInterface.
  */
class CpuBackend : public BackendInterface {
public:
	std::string name() const override;
	float* allocate(size_t num_elements) override;
	void free(float* ptr) override;
	void copy_to_device(float* dst, const float* src, size_t num_elements) override;
	void copy_to_host(float* dst, const float* src, size_t num_elements) override;
	void launch_kernel(const std::string& kernel_name,
		float* input1, float* input2, float* output, size_t size) override;
};