/**
 * @file    cuda_profiler.cpp
 * @brief   CUDA-specific profiler implementation for kernel execution timing.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Implements the CudaProfiler class for timing CUDA kernel execution using CUDA events.
 */

#include "cuda_profiler.hpp"

#include <cuda_runtime.h>

#include <fstream>
#include <nlohmann/json.hpp>

CudaProfiler::CudaProfiler() : last_time_ms(0.0f)
{
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);
}

CudaProfiler::~CudaProfiler()
{
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
}

void CudaProfiler::start()
{
	cudaEventRecord(start_event, 0);
}

void CudaProfiler::stop()
{
	cudaEventRecord(stop_event, 0);
	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&last_time_ms, start_event, stop_event);
}

double CudaProfiler::elapsed_time_ms() const
{
	return static_cast<double>(last_time_ms);
}

void CudaProfiler::save_to_json(const std::string& filename) const
{
	nlohmann::json j;
	j["cuda_kernel_execution_time_ms"] = last_time_ms;
	std::ofstream out(filename);
	out << j.dump(2);
}
