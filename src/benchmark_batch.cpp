#include "benchmark_utils.h"
#include "kernel_dispatch.h"
#include "input_generator.h"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <operation>\n";
        return 1;
    }

    std::string operation = argv[1];
    std::vector<int> sizes = { 1000, 10000, 100000, 1000000, 10000000 };
    int passes = 100;

    for (int N : sizes) {
        for (int pass = 1; pass <= passes; ++pass) {
            std::vector<float> a, b;
            generate_random_input(N, a, b);
            std::vector<float> c(N);

            BenchmarkResult result = dispatch_and_benchmark(operation, a.data(), b.data(), c.data(), N);

            std::cout << "[N = " << N << ", pass = " << pass << "] CPU: " << result.cpu_time
                << " ms, Global: " << result.gpu_global_time
                << " ms, Shared: " << result.gpu_shared_time
                << " ms, Float4: " << result.gpu_float4_time << " ms\n";

            append_result_to_csv("result.csv", operation, N, result);
        }
    }

    return 0;
}