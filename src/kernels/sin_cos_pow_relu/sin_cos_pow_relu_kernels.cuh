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
 * Applies sin, cos, pow, and ReLU operations elementwise on the inputs.
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 */
__global__ void sin_cos_pow_relu_global_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for sin_cos_pow_relu using shared memory.
 *
 * Applies sin, cos, pow, and ReLU operations elementwise on the inputs, with potential for shared memory optimization.
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 */
__global__ void sin_cos_pow_relu_shared_kernel(const float* a, const float* b, float* c, int N);

/**
 * @brief CUDA kernel for vectorized sin_cos_pow_relu operation using float4 types.
 *
 * Each thread processes four elements packed in float4.
 *
 * @param[in]  a  Pointer to the first input array (float4, global memory).
 * @param[in]  b  Pointer to the second input array (float4, global memory).
 * @param[out] c  Pointer to the output array (float4, global memory).
 * @param[in]  N  Number of float4 elements.
 */
__global__ void sin_cos_pow_relu_float4_kernel(const float4* a, const float4* b, float4* c, int N);