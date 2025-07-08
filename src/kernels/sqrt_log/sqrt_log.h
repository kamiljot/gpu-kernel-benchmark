/**
 * @file    sqrt_log.h
 * @brief   Host launchers for sqrt_log kernel variants (global, shared memory, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains host functions to launch various sqrt_log kernels for benchmarking.
 */

#pragma once

/**
 * @brief Runs the global memory sqrt_log kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_sqrt_log_global(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the shared memory sqrt_log kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_sqrt_log_shared(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the float4 vectorized sqrt_log kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_sqrt_log_float4(const float* a, const float* b, float* c, int N);