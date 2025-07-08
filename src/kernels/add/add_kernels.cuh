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
 * @brief CUDA kernel for basic element-wise addition using global memory.
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 */
__global__ void add_global_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for element-wise addition using shared memory for block-level efficiency.
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 */
__global__ void add_shared_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for vectorized element-wise addition using float4 types.
 *
 * @param[in]  a  Pointer to the first input array (float4, global memory).
 * @param[in]  b  Pointer to the second input array (float4, global memory).
 * @param[out] c  Pointer to the output array (float4, global memory).
 * @param[in]  N  Number of float4 elements.
 */
__global__ void add_float4_kernel(const float4* a, const float4* b, float4* c, int N);