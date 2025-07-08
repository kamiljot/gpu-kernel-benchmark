/**
 * @file    sqrt_log_kernels.cuh
 * @brief   Device kernel declarations for different sqrt_log kernel implementations (global, shared, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Declares CUDA kernels for the sqrt_log operation in global memory, shared memory, and float4 vectorized variants.
 */

#pragma once
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for elementwise sqrt_log operation using global memory.
 *
 * Each output is computed as sqrt(a[i]) + log(b[i] + 1e-6).
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 */
__global__ void sqrt_log_global_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for elementwise sqrt_log operation using shared memory.
 *
 * Loads data into shared memory before computing sqrt and log for improved memory access.
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 */
__global__ void sqrt_log_shared_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for vectorized sqrt_log operation using float4 types.
 *
 * Each thread processes four elements packed in float4.
 *
 * @param[in]  a  Pointer to the first input array (float4, global memory).
 * @param[in]  b  Pointer to the second input array (float4, global memory).
 * @param[out] c  Pointer to the output array (float4, global memory).
 * @param[in]  N  Number of float4 elements.
 */
__global__ void sqrt_log_float4_kernel(const float4* a, const float4* b, float4* c, int N);