#include "benchmark_utils.h"
#include "kernel_dispatch.h"
#include "input_generator.h"
#include <iostream>
#include <string>
#include <unordered_map>

/**
 * Entry point for single-pass benchmark.
 * Usage: ./gpu_kernel_benchmark --op <operation> --variant <variant> --input <input_file> --passes <num_passes>
 */
int main(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i][0] == '-' && argv[i + 1][0] != '-') {
            args[argv[i]] = argv[i + 1];
            ++i;
        }
    }

    std::string operation = args.count("--op") ? args["--op"] : "sqrt_log";
    std::string variant = args.count("--variant") ? args["--variant"] : "all";
    std::string input_path = args.count("--input") ? args["--input"] : "input_file";
    int passes = args.count("--passes") ? std::stoi(args["--passes"]) : 1;

    std::vector<float> a, b;
    if (!read_input_file(input_path, a, b)) {
        std::cout << "Input file not found. Generating random data...\n";
        int N = 1000000;
        generate_random_input(N, a, b);
        write_input_file(input_path, a, b);
    }

    std::vector<float> c(a.size());

    for (int i = 0; i < passes; ++i) {
        BenchmarkResult result = dispatch_and_benchmark(operation, variant, input_path, a.data(), b.data(), c.data(), static_cast<int>(a.size()));

        std::cout << "[N = " << a.size() << ", pass = " << (i + 1)
            << "] CPU: " << result.cpu_time
            << " ms, Global: " << result.gpu_global_time
            << " ms, Shared: " << result.gpu_shared_time
            << " ms, Float4: " << result.gpu_float4_time << " ms\n";

        append_result_to_csv("result.csv", operation, static_cast<int>(a.size()), result);
    }

    return 0;
}