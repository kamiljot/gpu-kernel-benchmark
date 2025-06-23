#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include "timer.hpp"
#include "gpu_memory_utils.h"
#include "gpu_benchmark_utils.h"
#include "benchmark_utils.h"

extern void cpu_add(const float*, const float*, float*, int);

int main(int argc, char** argv) {
    std::filesystem::create_directories("benchmarks");

    // Determine number of passes from CLI (default: 100)
    int passes = 100;
    if (argc > 1) {
        passes = std::atoi(argv[1]);
        if (passes <= 0) passes = 100;
    }

    std::vector<int> sizes = { 1000, 10'000, 100'000, 1'000'000, 10'000'000 };

    std::ofstream out("benchmarks/results_all.csv");
    out << "size,pass,cpu_time_ms,gpu_global_ms,gpu_shared_ms,gpu_float4_ms\n";

    for (int N : sizes) {
        if (N % 4 != 0) continue; // required for float4 compatibility

        for (int pass = 1; pass <= passes; ++pass) {
            std::vector<float> a(N), b(N), c_cpu(N), c_gpu(N);
            generate_input_data(a, b);

            // CPU baseline timing
            Timer timer;
            timer.start();
            cpu_add(a.data(), b.data(), c_cpu.data(), N);
            double cpu_time = timer.stop();

            // Allocate GPU buffers and copy data
            float* d_a, * d_b, * d_c;
            allocate_and_copy(a.data(), b.data(), &d_a, &d_b, &d_c, N);

            // Benchmark kernels
            float time_global = benchmark_global_kernel(d_a, d_b, d_c, N);
            float time_shared = benchmark_shared_kernel(d_a, d_b, d_c, N, 256);
            float time_float4 = benchmark_float4_kernel(a.data(), b.data(), N);

            cudaMemcpy(c_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
            free_device(d_a, d_b, d_c);

            // Print and record results
            std::cout << "[N = " << N << ", pass = " << pass
                << "] CPU: " << cpu_time << " ms, "
                << "Global: " << time_global << " ms, "
                << "Shared: " << time_shared << " ms, "
                << "Float4: " << time_float4 << " ms\n";

            write_result(out, N, pass, cpu_time, time_global, time_shared, time_float4);
        }
    }

    return 0;
}
