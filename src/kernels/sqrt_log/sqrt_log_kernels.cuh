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
 * Each thread computes: c[i] = sqrtf(a[i]) + logf(b[i] + 1e-6f).
 * The small constant 1e-6 prevents domain errors for non-positive inputs to log.
 *
 * @param[in]  a  Pointer to the first input array (device global memory).
 * @param[in]  b  Pointer to the second input array (device global memory).
 * @param[out] c  Pointer to the output array (device global memory).
 * @param[in]  N  Number of elements to process.
 */
__global__ void sqrt_log_global_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for elementwise sqrt_log operation using shared memory.
 *
 * Loads per-block tiles of data into shared memory before computing sqrt and log,
 * improving memory access patterns and reducing global memory traffic.
 *
 * @param[in]  a  Pointer to the first input array (device global memory).
 * @param[in]  b  Pointer to the second input array (device global memory).
 * @param[out] c  Pointer to the output array (device global memory).
 * @param[in]  N  Number of elements to process.
 */
__global__ void sqrt_log_shared_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for vectorized sqrt_log operation using float4 types.
 *
 * Each thread processes one float4 element (4 scalar floats), computing the sqrt_log
 * expression for all four components in a vectorized manner.
 *
 * @param[in]  a  Pointer to the first input array (device global memory, float4).
 * @param[in]  b  Pointer to the second input array (device global memory, float4).
 * @param[out] c  Pointer to the output array (device global memory, float4).
 * @param[in]  N  Number of float4 elements to process.
 */
__global__ void sqrt_log_float4_kernel(const float4* a, const float4* b, float4* c, int N);
