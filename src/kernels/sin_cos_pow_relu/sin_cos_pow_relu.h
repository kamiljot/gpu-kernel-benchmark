/**
 * @file    sin_cos_pow_relu.h
 * @brief   Host launchers for sin_cos_pow_relu kernel variants (global, shared memory, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains host functions to launch various sin_cos_pow_relu kernels for benchmarking.
 */

#pragma once

/**
 * @brief Runs the global memory sin_cos_pow_relu kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_sin_cos_pow_relu_global(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the shared memory sin_cos_pow_relu kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_sin_cos_pow_relu_shared(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the float4 vectorized sin_cos_pow_relu kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_sin_cos_pow_relu_float4(const float* a, const float* b, float* c, int N);