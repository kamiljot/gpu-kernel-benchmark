/**
 * @file    add.h
 * @brief   Host launchers for different add kernel variants (global, shared memory, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains host functions to launch various addition kernels for benchmarking.
 */

#pragma once

/**
 * @brief Runs the global memory add kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_add_global(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the shared memory add kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_add_shared(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the float4 vectorized add kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_add_float4(const float* a, const float* b, float* c, int N);