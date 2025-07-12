/**
 * @file    benchmark_cuda.cpp
 * @brief   CUDA benchmark implementation (modular).
 * @author  Kamil J.
 * @date    2025-07-12
 */
#include "benchmark/benchmark_cuda.hpp"

#include "backend/gpu_timing.hpp"

/**
 * @brief   Runs a full CUDA kernel benchmark pass (with timings).
 */
CudaBenchmarkResult run_cuda_benchmark(BackendInterface* backend, const std::string& kernel_name,
                                       const std::vector<float>& in1, const std::vector<float>& in2,
                                       std::vector<float>& out, size_t size)
{
    float* d_in1 = backend->allocate(size);
    float* d_in2 = backend->allocate(size);
    float* d_out = backend->allocate(size);

    GpuTiming timing{};
    backend->copy_to_device(d_in1, in1.data(), size);
    backend->copy_to_device(d_in2, in2.data(), size);

    backend->launch_kernel(kernel_name, d_in1, d_in2, d_out, in1.data(), in2.data(), out.data(), size, &timing);
    backend->copy_to_host(out.data(), d_out, size);

    backend->free(d_in1);
    backend->free(d_in2);
    backend->free(d_out);

    CudaBenchmarkResult result;
    result.transfer_in_ms = timing.transfer_in_ms;
    result.kernel_ms = timing.kernel_ms;
    result.transfer_out_ms = timing.transfer_out_ms;
    return result;
}