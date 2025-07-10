/**
 * @file    cuda_backend.hpp
 * @brief   CUDA backend implementation for modular GPU kernel benchmark.
 * @author  Kamil J.
 * @date    2025-07-09
 */

#pragma once
#include <string>
#include <vector>
class Kernel;  ///< Forward declaration

/**
 * @class   CudaBackend
 * @brief   Backend implementation for running GPU kernels using CUDA.
 */
class CudaBackend
{
public:
	CudaBackend();
	~CudaBackend();

	std::string name() const;
	void initialize();
	void load_kernel(const std::string& kernel_name);
	void launch(Kernel* kernel, const std::vector<float>& input_a, const std::vector<float>& input_b,
		std::vector<float>& output);
	double get_last_execution_time() const;

private:
	double last_execution_time_ms;
};