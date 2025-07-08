/**
 * @file    test_add_correctness.cpp
 * @brief   Quick correctness test: compares CPU and GPU output for "add" operation.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../include/kernel_dispatch.h"

int main()
{
    const int N = 1000;
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c_cpu(N), c_gpu(N);

    // CPU
    BenchmarkResult r_cpu = dispatch_and_benchmark("add", a.data(), b.data(), c_cpu.data(), N, "cpu");
    // GPU
    BenchmarkResult r_gpu = dispatch_and_benchmark("add", a.data(), b.data(), c_gpu.data(), N, "global");

    // Compare results (with tolerance)
    double max_err = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double err = std::abs(static_cast<double>(c_cpu[i] - c_gpu[i]));
        if (err > max_err) max_err = err;
    }

    std::cout << "Max CPU vs GPU error: " << max_err << std::endl;
    assert(max_err < 1e-5);  // Floats tolerance
    std::cout << "Test PASSED.\n";
    return 0;
}