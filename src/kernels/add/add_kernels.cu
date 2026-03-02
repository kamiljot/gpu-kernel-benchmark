/**
 * @file    add_kernels.cu
 * @brief   CUDA kernel implementations for the "add" operation: global memory, shared memory, and float4 variants.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains CUDA kernel implementations for elementwise addition in global memory, shared memory, and float4 vectorized
 * form.
 */


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
    extern __shared__ float smem[];
    float* s_a = smem;
    float* s_b = smem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    s_a[t] = (idx < N) ? a[idx] : 0.0f;
    s_b[t] = (idx < N) ? b[idx] : 0.0f;

    __syncthreads();

    if (idx < N) c[idx] = s_a[t] + s_b[t];
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