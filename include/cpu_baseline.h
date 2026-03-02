/**
 * @file    cpu_baseline.h
 * @brief   Reference CPU implementations for benchmarking purposes.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains CPU reference versions of operations for validating GPU kernel correctness and performance.
 */

#pragma once

/**
 * @brief Performs elementwise addition of two float arrays on the CPU and measures elapsed time.
 *
 * This reference implementation is used for correctness validation and performance comparison
 * against GPU kernel variants. The function iterates over the input arrays and writes the
 * sum into the output array.
 *
 * @param[in]  a  Pointer to the first input array (size N).
 * @param[in]  b  Pointer to the second input array (size N).
 * @param[out] c  Pointer to the output array (size N).
 * @param[in]  N  Number of elements to process.
 * @return        Elapsed time in milliseconds.
 */
float run_cpu_add(const float* a, const float* b, float* c, int N);

/**
 * @brief Computes elementwise sqrt(a[i]) + log(b[i] + 1e-6) on the CPU and measures elapsed time.
 *
 * The routine implements the same arithmetic as the GPU sqrt_log kernels and is used to
 * validate correctness. The small constant 1e-6 is added inside the logarithm to avoid
 * domain errors for non-positive inputs.
 *
 * @param[in]  a  Pointer to the first input array (size N).
 * @param[in]  b  Pointer to the second input array (size N).
 * @param[out] c  Pointer to the output array (size N).
 * @param[in]  N  Number of elements to process.
 * @return        Elapsed time in milliseconds.
 */
float run_cpu_sqrt_log(const float* a, const float* b, float* c, int N);

/**
 * @brief Computes elementwise relu(pow(sin(a[i]) + cos(b[i]), 2.0)) on the CPU and measures elapsed time.
 *
 * The function performs the following for each element:
 *   tmp = sin(a[i]) + cos(b[i]);
 *   out = pow(tmp, 2.0);
 *   c[i] = max(0.0f, out);
 *
 * This implementation is the reference for validating the corresponding GPU kernels.
 *
 * @param[in]  a  Pointer to the first input array (size N).
 * @param[in]  b  Pointer to the second input array (size N).
 * @param[out] c  Pointer to the output array (size N).
 * @param[in]  N  Number of elements to process.
 * @return        Elapsed time in milliseconds.
 */
float run_cpu_sin_cos_pow_relu(const float* a, const float* b, float* c, int N);
