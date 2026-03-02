/**
 * @file    sqrt_log_kernels.cu
 * @brief   CUDA kernel implementations for the sqrt_log operation: global memory, shared memory, and float4 variants.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains CUDA kernel implementations for the sqrt_log operation,
 * including global memory, shared memory, and float4 vectorized variants.
 */

#include <math.h>

#include "sqrt_log_kernels.cuh"

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
__global__ void sqrt_log_global_kernel(const float* a, const float* b, float* c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        c[i] = sqrtf(a[i]) + logf(b[i] + 1e-6f);
    }
}

/**
 * @brief CUDA kernel for elementwise sqrt_log operation using shared memory.
 *
 * Loads blocks of input data into shared memory before computing sqrt and log for improved memory access.
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 */
__global__ void sqrt_log_shared_kernel(const float* a, const float* b, float* c, int N)
{
    extern __shared__ float smem[];
    float* s_a = smem;
    float* s_b = smem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    s_a[t] = (idx < N) ? a[idx] : 0.0f;
    s_b[t] = (idx < N) ? b[idx] : 0.0f;
    
    __syncthreads();

    if (idx < N) c[idx] = sqrtf(s_a[t]) + logf(s_b[t] + 1e-6f);
}

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
__global__ void sqrt_log_float4_kernel(const float4* a, const float4* b, float4* c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float4 va = a[i];
        float4 vb = b[i];
        c[i] = make_float4(sqrtf(va.x) + logf(vb.x + 1e-6f), sqrtf(va.y) + logf(vb.y + 1e-6f),
                           sqrtf(va.z) + logf(vb.z + 1e-6f), sqrtf(va.w) + logf(vb.w + 1e-6f));
    }
}