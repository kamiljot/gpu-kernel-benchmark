/**
 * @file    benchmark_runner.cpp
 * @brief   Main benchmark runner for modular GPU kernel benchmarking framework.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Usage example:
 *   ./gpu-kernel-benchmark --op add --variant all --size 1048576 --passes 10 --csv results.csv
 */

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "backend/backend_registry.hpp"
#include "backend/cuda_backend.hpp"
#include "kernels/kernel_registry.hpp"
#include "utils/csv_logger.hpp"
#include "utils/time_utils.hpp"

struct Args
{
    std::string backend = "cpu";
    std::string op = "add";
    std::string variant = "all";
    size_t size = 1024 * 1024;
    int passes = 10;
    std::string csv_file = "results.csv";
};

Args parse_args(int argc, char* argv[])
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc)
            args.backend = argv[++i];
        else if (a == "--op" && i + 1 < argc)
            args.op = argv[++i];
        else if (a == "--variant" && i + 1 < argc)
            args.variant = argv[++i];
        else if (a == "--size" && i + 1 < argc)
            args.size = std::stoul(argv[++i]);
        else if (a == "--passes" && i + 1 < argc)
            args.passes = std::stoi(argv[++i]);
        else if (a == "--csv" && i + 1 < argc)
            args.csv_file = argv[++i];
    }
    return args;
}

std::vector<float> generate_input(size_t size)
{
    std::vector<float> data(size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) v = dist(rng);
    return data;
}

extern void force_backend_registration();
extern void force_kernel_registration();

int main(int argc, char* argv[])
{
    force_backend_registration();
    force_kernel_registration();
    Args args = parse_args(argc, argv);

    auto backend = BackendRegistry::instance().create(args.backend);
    if (!backend)
    {
        std::cerr << "[ERROR] Backend not found: " << args.backend << std::endl;
        return 1;
    }

    // List available kernels
    auto all_kernels = KernelRegistry::instance().available_kernels();
    std::vector<std::string> selected_kernels;
    if (args.variant == "all")
    {
        for (const auto& name : all_kernels)
        {
            if (name.find(args.op) == 0) selected_kernels.push_back(name);
        }
    }
    else
    {
        std::string full_name = args.op + "_" + args.variant;
        if (std::find(all_kernels.begin(), all_kernels.end(), full_name) != all_kernels.end())
            selected_kernels.push_back(full_name);
        else
        {
            std::cerr << "[ERROR] Kernel variant not found: " << full_name << std::endl;
            return 1;
        }
    }

    std::cout << "[INFO] Selected kernels:\n";
    for (const auto& k : selected_kernels) std::cout << "  - " << k << "\n";

    // Prepare input data
    auto in1 = generate_input(args.size);
    auto in2 = generate_input(args.size);
    std::vector<float> out(args.size);

    CsvLogger logger(args.csv_file, {"kernel", "size", "pass", "time_ms"});

    for (const auto& kernel_name : selected_kernels)
    {
        std::cout << "[INFO] Benchmarking kernel: " << kernel_name << "\n";
        for (int p = 0; p < args.passes; ++p)
        {
            // Prepare input/output (host & device)
            auto in1 = generate_input(args.size);
            auto in2 = generate_input(args.size);
            std::vector<float> out(args.size, 0.0f);

            if (args.backend == "cpu")
            {
                auto kernel = KernelRegistry::instance().create(kernel_name);
                CpuTimer timer;
                timer.start();
                kernel->run(const_cast<float*>(in1.data()), const_cast<float*>(in2.data()), out.data(), args.size);
                timer.stop();
                double elapsed = timer.elapsed_ms();
                logger.write_row({kernel_name, std::to_string(args.size), std::to_string(p), std::to_string(elapsed)});
                std::cout << "  Pass " << p << ": " << elapsed << " ms\n";
            }
            else if (args.backend == "cuda")
            {
                float* d_in1 = backend->allocate(args.size);
                float* d_in2 = backend->allocate(args.size);
                float* d_out = backend->allocate(args.size);

                GpuTiming timing;
                backend->copy_to_device(d_in1, in1.data(), args.size);
                backend->copy_to_device(d_in2, in2.data(), args.size);

                backend->launch_kernel(kernel_name, d_in1, d_in2, d_out, in1.data(), in2.data(), out.data(), args.size,
                                       &timing);
                backend->copy_to_host(out.data(), d_out, args.size);

                std::cout << "  [INFO] transfer_in: " << timing.transfer_in_ms << " ms, "
                          << "kernel: " << timing.kernel_ms << " ms, "
                          << "transfer_out: " << timing.transfer_out_ms << " ms\n";

                backend->free(d_in1);
                backend->free(d_in2);
                backend->free(d_out);
            }
        }
    }
    std::cout << "[INFO] Benchmark finished. Results in: " << args.csv_file << "\n";
    return 0;
}