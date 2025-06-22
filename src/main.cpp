#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <filesystem>
#include "timer.hpp"

extern void cpu_add(const float*, const float*, float*, int);
extern "C" void gpu_math(const float*, const float*, float*, int, float*);

int main() {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories("benchmarks");

    // Input sizes to benchmark
    std::vector<int> sizes = { 1000, 10'000, 100'000, 1'000'000, 10'000'000 };
    const int passes = 10; // number of repetitions per size

    std::ofstream out("benchmarks/results_multi.csv");
    out << "size,pass,cpu_time_ms,gpu_time_ms\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);

    for (int N : sizes) {
        for (int pass = 1; pass <= passes; ++pass) {
            // Allocate and fill input data
            std::vector<float> a(N), b(N), c_cpu(N), c_gpu(N);
            for (int i = 0; i < N; ++i) {
                a[i] = dist(rng);
                b[i] = dist(rng);
            }

            // CPU timing
            Timer timer;
            timer.start();
            cpu_add(a.data(), b.data(), c_cpu.data(), N);
            double cpu_time = timer.stop();

            // GPU timing
            float gpu_kernel_time = 0.0f;
            gpu_math(a.data(), b.data(), c_gpu.data(), N, &gpu_kernel_time);

            std::cout << "[N = " << N << ", pass = " << pass << "] CPU: " << cpu_time << " ms, GPU: " << gpu_kernel_time << " ms\n";
            out << N << "," << pass << "," << cpu_time << "," << gpu_kernel_time << "\n";
        }
    }

    return 0;
}