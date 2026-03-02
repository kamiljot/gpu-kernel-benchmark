/**
 * @file    sin_cos_pow_relu_kernels.cuh
 * @brief   Device kernel declarations for sin_cos_pow_relu kernel (global, shared, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Declares CUDA kernels for the sin_cos_pow_relu operation, covering global memory, shared memory, and float4 variants.
 */

#pragma once
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for sin_cos_pow_relu using global memory.
 *
 * Each thread computes: c[i] = max(0.0f, pow(sin(a[i]) + cos(b[i]), 2.0f)).
 * The operation combines trigonometric functions, exponentiation, and ReLU activation.
 *
 * @param[in]  a  Pointer to the first input array (device global memory).
 * @param[in]  b  Pointer to the second input array (device global memory).
 * @param[out] c  Pointer to the output array (device global memory).
 * @param[in]  N  Number of elements to process.
 */
__global__ void sin_cos_pow_relu_global_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for sin_cos_pow_relu using shared memory.
 *
 * Loads per-block tiles into shared memory before applying the sin_cos_pow_relu expression,
 * improving memory access patterns for the complex arithmetic operations.
 *
 * @param[in]  a  Pointer to the first input array (device global memory).
 * @param[in]  b  Pointer to the second input array (device global memory).
 * @param[out] c  Pointer to the output array (device global memory).
 * @param[in]  N  Number of elements to process.
 */
__global__ void sin_cos_pow_relu_shared_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for vectorized sin_cos_pow_relu operation using float4 types.
 *
 * Each thread processes one float4 element (4 scalar floats), computing the sin_cos_pow_relu
 * expression for all four components in a vectorized manner.
 *
 * @param[in]  a  Pointer to the first input array (device global memory, float4).
 * @param[in]  b  Pointer to the second input array (device global memory, float4).
 * @param[out] c  Pointer to the output array (device global memory, float4).
 * @param[in]  N  Number of float4 elements to process.
 */
__global__ void sin_cos_pow_relu_float4_kernel(const float4* a, const float4* b, float4* c, int N);
