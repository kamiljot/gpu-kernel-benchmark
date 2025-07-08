/**
 * @file    add_kernels.cu
 * @brief   CUDA kernel implementations for the "add" operation: global memory, shared memory, and float4 variants.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains CUDA kernel implementations for elementwise addition in global memory, shared memory, and float4 vectorized
 * form.
 */

#pragma once

#include "add_kernels.cuh"

/**
 * @brief CUDA kernel for elementwise addition using global memory.
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 */
__global__ void add_global_kernel(const float* a, const float* b, float* c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief CUDA kernel for elementwise addition using shared memory for improved memory access efficiency.
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 *
 * Loads a block of elements into shared memory before computing the addition.
 */
__global__ void add_shared_kernel(const float* a, const float* b, float* c, int N)
{
    __shared__ float s_a[256];
    __shared__ float s_b[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load elements into shared memory, pad with zero if out of range
    s_a[tid] = (idx < N) ? a[idx] : 0.0f;
    s_b[tid] = (idx < N) ? b[idx] : 0.0f;

    __syncthreads();  // Synchronize to ensure all loads complete

    if (idx < N)
    {
        c[idx] = s_a[tid] + s_b[tid];
    }
}

/**
 * @brief CUDA kernel for vectorized elementwise addition using float4 types.
 *
 * @param[in]  a  Pointer to the first input array (float4, global memory).
 * @param[in]  b  Pointer to the second input array (float4, global memory).
 * @param[out] c  Pointer to the output array (float4, global memory).
 * @param[in]  N  Number of float4 elements.
 */
__global__ void add_float4_kernel(const float4* a, const float4* b, float4* c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float4 va = a[i];
        float4 vb = b[i];
        c[i] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }
}