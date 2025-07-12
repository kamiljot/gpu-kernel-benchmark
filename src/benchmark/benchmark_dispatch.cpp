/**
 * @file    benchmark_dispatch.cpp
 * @brief   Benchmark execution for all backends/kernels.
 * @author  Kamil J.
 * @date    2025-07-12
 */

#include "benchmark/benchmark_dispatch.hpp"

#include <iostream>

#include "backend/backend_registry.hpp"
#include "benchmark/benchmark_cpu.hpp"
#include "benchmark/benchmark_cuda.hpp"
#include "benchmark/generators.hpp"
#include "utils/csv_logger.hpp"

void dispatch_benchmarks(const Args& args, const std::vector<std::string>& backends,
                         const std::vector<std::string>& kernels, CsvLogger& logger)
{
    for (const auto& backend_name : backends)
    {
        auto backend = BackendRegistry::instance().create(backend_name);
        if (!backend)
        {
            std::cerr << "[ERROR] Backend not found: " << backend_name << std::endl;
            continue;
        }
        std::cout << "[INFO] Benchmarking on backend: " << backend_name << std::endl;

        for (const auto& kernel_name : kernels)
        {
            std::cout << "[INFO]  Kernel: " << kernel_name << std::endl;
            for (int p = 0; p < args.passes; ++p)
            {
                auto in1 = generate_input(args.size);
                auto in2 = generate_input(args.size);
                std::vector<float> out(args.size, 0.0f);

                if (backend_name == "cpu")
                {
                    CpuBenchmarkResult result = run_cpu_benchmark(kernel_name, in1, in2, out, args.size);
                    logger.write_row({backend_name, kernel_name, std::to_string(args.size), std::to_string(p),
                                      std::to_string(result.elapsed_ms), "", "", ""});
                }
                else if (backend_name == "cuda")
                {
                    CudaBenchmarkResult result =
                        run_cuda_benchmark(backend.get(), kernel_name, in1, in2, out, args.size);
                    logger.write_row({backend_name, kernel_name, std::to_string(args.size), std::to_string(p), "",
                                      std::to_string(result.transfer_in_ms), std::to_string(result.kernel_ms),
                                      std::to_string(result.transfer_out_ms)});
                }
            }
        }
    }
}