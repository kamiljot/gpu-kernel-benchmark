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
 * @brief Performs elementwise addition of two arrays on the CPU.
 *
 * @param[in]  a  First input array.
 * @param[in]  b  Second input array.
 * @param[out] c  Output array.
 * @param[in]  N  Number of elements.
 * @return        Elapsed time in milliseconds.
 */
float run_cpu_add(const float* a, const float* b, float* c, int N);

/**
 * @brief Performs elementwise sqrt and log operations on arrays on the CPU.
 *
 * @param[in]  a  First input array.
 * @param[in]  b  Second input array.
 * @param[out] c  Output array.
 * @param[in]  N  Number of elements.
 * @return        Elapsed time in milliseconds.
 */
float run_cpu_sqrt_log(const float* a, const float* b, float* c, int N);

/**
 * @brief Performs elementwise sin, cos, pow, and ReLU operations on arrays on the CPU.
 *
 * @param[in]  a  First input array.
 * @param[in]  b  Second input array.
 * @param[out] c  Output array.
 * @param[in]  N  Number of elements.
 * @return        Elapsed time in milliseconds.
 */
float run_cpu_sin_cos_pow_relu(const float* a, const float* b, float* c, int N);