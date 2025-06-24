#include "benchmark_utils.h"
#include "kernel_dispatch.h"
#include "input_generator.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string operation = "add";
    std::string input_path = "input_file.bin";

    if (argc >= 2) operation = argv[1];
    if (argc >= 3) input_path = argv[2];

    std::vector<float> a, b;
    if (!read_input_file(input_path, a, b)) {
        std::cout << "Input file not found. Generating random data...\n";
        int N = 1000000;
        generate_random_input(N, a, b);
        write_input_file(input_path, a, b);
    }

    std::vector<float> c(a.size());

    BenchmarkResult result = dispatch_and_benchmark(operation, a.data(), b.data(), c.data(), static_cast<int>(a.size()));

    std::cout << "[N = " << a.size() << "] CPU: " << result.cpu_time
        << " ms, Global: " << result.gpu_global_time
        << " ms, Shared: " << result.gpu_shared_time
        << " ms, Float4: " << result.gpu_float4_time << " ms\n";

    append_result_to_csv("result.csv", operation, static_cast<int>(a.size()), result);
    return 0;
}