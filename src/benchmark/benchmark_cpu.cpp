/**
 * @file    benchmark_cpu.cpp
 * @brief   CPU benchmark implementation (modular).
 * @author  Kamil J.
 * @date    2025-07-12
 */
#include "benchmark/benchmark_cpu.hpp"

#include <cassert>

#include "kernels/kernel_registry.hpp"
#include "utils/time_utils.hpp"

CpuBenchmarkResult run_cpu_benchmark(const std::string& kernel_name, const std::vector<float>& in1,
                                     const std::vector<float>& in2, std::vector<float>& out, size_t size)
{
    auto kernel = KernelRegistry::instance().create(kernel_name);
    assert(kernel && "Kernel not found!");

    CpuTimer timer;
    timer.start();
    kernel->run(const_cast<float*>(in1.data()), const_cast<float*>(in2.data()), out.data(), size);
    timer.stop();

    CpuBenchmarkResult result;
    result.elapsed_ms = timer.elapsed_ms();
    return result;
}