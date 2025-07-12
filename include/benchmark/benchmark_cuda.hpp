/**
 * @file    benchmark_cuda.hpp
 * @brief   Standalone function for running CUDA benchmarks (modular interface).
 * @author  Kamil J.
 * @date    2025-07-12
 */
#pragma once

#include <string>
#include <vector>

#include "backend/backend_interface.hpp"
#include "backend/gpu_timing.hpp"

/**
 * @struct CudaBenchmarkResult
 * @brief  Holds the result of a single CUDA benchmark pass (transfer + kernel timings).
 */
struct CudaBenchmarkResult
{
    float transfer_in_ms;   ///< Host-to-device transfer time (ms).
    float kernel_ms;        ///< Kernel execution time (ms).
    float transfer_out_ms;  ///< Device-to-host transfer time (ms).
};

/**
 * @brief  Runs a CUDA kernel benchmark for given inputs, measuring transfers and kernel.
 * @param  backend      Pointer to CUDA backend (already constructed).
 * @param  kernel_name  Name of the registered kernel.
 * @param  in1          First input vector (host memory).
 * @param  in2          Second input vector (host memory).
 * @param  out          Output vector (host memory, will be overwritten).
 * @param  size         Number of elements to process.
 * @return Struct with detailed timing results.
 */
CudaBenchmarkResult run_cuda_benchmark(BackendInterface* backend, const std::string& kernel_name,
                                       const std::vector<float>& in1, const std::vector<float>& in2,
                                       std::vector<float>& out, size_t size);