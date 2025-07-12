/**
 * @file    dispatch_benchmarks.cpp
 * @brief   Functions for dispatching and reporting benchmarks for multiple backends/kernels (implementation).
 * @author  Kamil J.
 * @date    2025-07-12
 */

#include "benchmark/dispatch_benchmarks.hpp"

#include <iostream>
#include <map>

#include "backend/backend_interface.hpp"
#include "backend/backend_registry.hpp"
#include "benchmark/benchmark_cpu.hpp"
#include "benchmark/benchmark_cuda.hpp"
#include "benchmark/generators.hpp"
#include "utils/csv_logger.hpp"

void dispatch_benchmarks(const Args& args, const std::vector<std::string>& backends,
                         const std::vector<std::string>& kernels, CsvLogger& logger)
{
    struct CpuStats
    {
        double sum = 0.0;
        int count = 0;
    };
    struct GpuStats
    {
        double transfer_in = 0.0, kernel = 0.0, transfer_out = 0.0;
        int count = 0;
    };

    std::map<std::string, CpuStats> cpu_stats;
    std::map<std::string, GpuStats> gpu_stats;

    for (const auto& backend_name : backends)
    {
        auto backend = BackendRegistry::instance().create(backend_name);
        if (!backend)
        {
            std::cerr << "[ERROR] Backend not found: " << backend_name << std::endl;
            continue;
        }

        for (const auto& kernel_name : kernels)
        {
            for (int p = 0; p < args.passes; ++p)
            {
                auto in1 = generate_input(args.size);
                auto in2 = generate_input(args.size);
                std::vector<float> out(args.size, 0.0f);

                if (backend_name == "cpu")
                {
                    CpuBenchmarkResult result = run_cpu_benchmark(kernel_name, in1, in2, out, args.size);
                    cpu_stats[kernel_name].sum += result.elapsed_ms;
                    cpu_stats[kernel_name].count++;
                    logger.write_row({backend_name, kernel_name, std::to_string(args.size), std::to_string(p),
                                      std::to_string(result.elapsed_ms), "", "", ""});
                    std::cout << "[CPU] " << kernel_name << " pass " << p << " : " << result.elapsed_ms << " ms\n";
                }
                else if (backend_name == "cuda")
                {
                    CudaBenchmarkResult result =
                        run_cuda_benchmark(backend.get(), kernel_name, in1, in2, out, args.size);
                    auto& s = gpu_stats[kernel_name];
                    s.transfer_in += result.transfer_in_ms;
                    s.kernel += result.kernel_ms;
                    s.transfer_out += result.transfer_out_ms;
                    s.count++;
                    logger.write_row({backend_name, kernel_name, std::to_string(args.size), std::to_string(p), "",
                                      std::to_string(result.transfer_in_ms), std::to_string(result.kernel_ms),
                                      std::to_string(result.transfer_out_ms)});
                    std::cout << "[CUDA] " << kernel_name << " pass " << p << " : "
                              << "[H2D] " << result.transfer_in_ms << " ms, [Kernel] " << result.kernel_ms
                              << " ms, [D2H] " << result.transfer_out_ms << " ms\n";
                }
            }
        }
    }

    std::cout << "\n[SUMMARY] Average times per kernel:\n";
    for (const auto& [kernel, stat] : cpu_stats)
        if (stat.count > 0) std::cout << "  [CPU]   " << kernel << " : " << (stat.sum / stat.count) << " ms\n";
    for (const auto& [kernel, stat] : gpu_stats)
        if (stat.count > 0)
            std::cout << "  [CUDA]  " << kernel << " : [H2D] " << (stat.transfer_in / stat.count) << " ms, [Kernel] "
                      << (stat.kernel / stat.count) << " ms, [D2H] " << (stat.transfer_out / stat.count) << " ms\n";
}