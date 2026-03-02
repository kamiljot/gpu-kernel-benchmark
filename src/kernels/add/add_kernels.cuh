/**
 * @file    add_kernels.cuh
 * @brief   Device kernel declarations for different add kernel implementations (global, shared, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Declares CUDA kernels for elementwise addition using various memory layouts and optimizations.
 */

#pragma once
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for elementwise addition using global memory.
 *
 * Each thread computes one element: c[i] = a[i] + b[i].
 * The kernel reads directly from global memory and writes the result back to global memory.
 *
 * @param[in]  a  Pointer to the first input array (device global memory).
 * @param[in]  b  Pointer to the second input array (device global memory).
 * @param[out] c  Pointer to the output array (device global memory).
 * @param[in]  N  Number of elements to process.
 */
__global__ void add_global_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for elementwise addition using shared memory for improved memory locality.
 *
 * The kernel stages per-block tiles of input data into shared memory before performing the
 * addition, reducing global memory traffic and improving cache efficiency.
 *
 * @param[in]  a  Pointer to the first input array (device global memory).
 * @param[in]  b  Pointer to the second input array (device global memory).
 * @param[out] c  Pointer to the output array (device global memory).
 * @param[in]  N  Number of elements to process.
 */
__global__ void add_shared_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for vectorized elementwise addition using float4 types.
 *
 * Each thread processes one float4 element (4 floats), improving memory throughput via
 * wider memory transactions.
 *
 * @param[in]  a  Pointer to the first input array (device global memory, float4).
 * @param[in]  b  Pointer to the second input array (device global memory, float4).
 * @param[out] c  Pointer to the output array (device global memory, float4).
 * @param[in]  N  Number of float4 elements to process.
 */
__global__ void add_float4_kernel(const float4* a, const float4* b, float4* c, int N);
