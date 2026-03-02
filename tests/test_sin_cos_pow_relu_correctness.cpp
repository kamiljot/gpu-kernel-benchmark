/**
 * @file    test_sin_cos_pow_relu_correctness.cpp
 * @brief   Correctness test: compares CPU baseline against GPU kernels for sin_cos_pow_relu operation.
 * @author  Kamil J.
 * @date    2025-07-07
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../include/cpu_baseline.h"
#include "../src/kernels/sin_cos_pow_relu/sin_cos_pow_relu.h"

int main()
{
    const int N = 100000;
    std::vector<float> a(N), b(N);
    
    // Initialize with values that will produce interesting results
    for (int i = 0; i < N; ++i)
    {
        a[i] = static_cast<float>(i % 100) * 0.1f - 5.0f;  // Range: -5 to +5
        b[i] = static_cast<float>((i * 7) % 100) * 0.1f - 5.0f;
    }

    std::vector<float> c_cpu(N), c_gpu_global(N), c_gpu_shared(N), c_gpu_float4(N);

    // CPU baseline
    run_cpu_sin_cos_pow_relu(a.data(), b.data(), c_cpu.data(), N);
    
    // GPU variants
    run_sin_cos_pow_relu_global(a.data(), b.data(), c_gpu_global.data(), N);
    run_sin_cos_pow_relu_shared(a.data(), b.data(), c_gpu_shared.data(), N);
    run_sin_cos_pow_relu_float4(a.data(), b.data(), c_gpu_float4.data(), N);

    // Compare results
    auto compare = [](const std::vector<float>& ref, const std::vector<float>& test, const char* label) {
        double max_err = 0.0;
        int mismatches = 0;
        for (size_t i = 0; i < ref.size(); ++i)
        {
            double err = std::abs(static_cast<double>(ref[i] - test[i]));
            if (err > max_err) max_err = err;
            if (err > 1e-4) mismatches++;
        }
        std::cout << label << " - Max error: " << max_err << ", Mismatches: " << mismatches << std::endl;
        assert(max_err < 1e-4);
        assert(mismatches == 0);
    };

    compare(c_cpu, c_gpu_global, "Global");
    compare(c_cpu, c_gpu_shared, "Shared");
    compare(c_cpu, c_gpu_float4, "Float4");

    std::cout << "All sin_cos_pow_relu tests PASSED.\n";
    return 0;
}
