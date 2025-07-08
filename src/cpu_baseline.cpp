/**
 * @file    cpu_baseline.cpp
 * @brief   Reference CPU implementations for benchmarking and validation.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Implements baseline CPU versions of mathematical operations for validation and performance comparison.
 */

#include "cpu_baseline.h"

#include <chrono>
#include <cmath>

/**
 * @brief Performs elementwise addition of two arrays on the CPU and measures elapsed time.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Elapsed time in milliseconds.
 */
float run_cpu_add(const float* a, const float* b, float* c, int N)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

/**
 * @brief Performs elementwise sqrt and log operations on arrays on the CPU and measures elapsed time.
 *
 * Each output is computed as sqrt(a[i]) + log(b[i] + 1e-6).
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Elapsed time in milliseconds.
 */
float run_cpu_sqrt_log(const float* a, const float* b, float* c, int N)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) c[i] = std::sqrt(a[i]) + std::log(b[i] + 1e-6f);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

/**
 * @brief Performs elementwise sin, cos, pow, and ReLU operations on arrays on the CPU and measures elapsed time.
 *
 * For each element: out = relu(pow(sin(a[i]) + cos(b[i]), 2.0)).
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Elapsed time in milliseconds.
 */
float run_cpu_sin_cos_pow_relu(const float* a, const float* b, float* c, int N)
{
    auto relu = [](float x) { return x > 0.0f ? x : 0.0f; };

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i)
    {
        float val = std::sin(a[i]) + std::cos(b[i]);
        val = std::pow(val, 2.0f);
        c[i] = relu(val);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> diff = end - start;

    return diff.count();
}