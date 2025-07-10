/**
 * @file    cuda_profiler.hpp
 * @brief   CUDA-specific profiler for kernel execution timing.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Declares the CudaProfiler class for timing CUDA kernel execution.
 */

#pragma once

#include <cuda_runtime.h>

#include <string>

#include "profiler.hpp"

 /**
  * @class   CudaProfiler
  * @brief   Profiler implementation using CUDA events.
  */
class CudaProfiler : public Profiler
{
public:
	CudaProfiler();
	~CudaProfiler() override;

	void start() override;
	void stop() override;
	double elapsed_time_ms() const override;
	void save_to_json(const std::string& filename) const override;

private:
	cudaEvent_t start_event;
	cudaEvent_t stop_event;
	float last_time_ms;
};