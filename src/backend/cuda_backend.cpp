/**
 * @file    cuda_backend.cpp
 * @brief   CUDA backend implementation for modular GPU kernel benchmark.
 * @author  Kamil J.
 * @date    2025-07-09
 */

#include "cuda_backend.hpp"

#include <iostream>

#include "kernels/kernel.hpp"  // Pe³ny include TYLKO w .cpp!

CudaBackend::CudaBackend() : last_execution_time_ms(0.0)
{
}

CudaBackend::~CudaBackend()
{
}

std::string CudaBackend::name() const
{
	return "cuda";
}

void CudaBackend::initialize()
{
	std::cout << "[CudaBackend] Initialized CUDA device." << std::endl;
}

void CudaBackend::load_kernel(const std::string& kernel_name)
{
	std::cout << "[CudaBackend] Loaded kernel: " << kernel_name << std::endl;
}

void CudaBackend::launch(Kernel* kernel, const std::vector<float>& input_a, const std::vector<float>& input_b,
	std::vector<float>& output)
{
	// (Opcjonalnie: tutaj mierzenie czasu)
	kernel->launch(input_a, input_b, output);
	last_execution_time_ms = 1.23;  // Tu wpisz realny pomiar czasu
}

double CudaBackend::get_last_execution_time() const
{
	return last_execution_time_ms;
}
