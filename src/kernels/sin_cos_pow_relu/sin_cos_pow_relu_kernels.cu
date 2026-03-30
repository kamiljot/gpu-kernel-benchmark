/**
 * @file    sin_cos_pow_relu_kernels.cu
 * @brief   CUDA kernel implementations for sin_cos_pow_relu (global, shared, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains CUDA kernel implementations for the sin_cos_pow_relu operation,
 * including global memory, shared memory, and float4 vectorized variants.
 */

#include <cmath>

#include "sin_cos_pow_relu_kernels.cuh"

/**
 * @brief Device function for ReLU activation.
 *
 * @param[in] x  Input value.
 * @return       Output after applying ReLU (max(0, x)).
 */
__device__ float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

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
__global__ void sin_cos_pow_relu_global_kernel(const float* a, const float* b, float* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float val = sinf(a[idx]) + cosf(b[idx]);
        val = powf(val, 2.0f);
        c[idx] = relu(val);
    }
}

/**
 * @brief CUDA kernel for sin_cos_pow_relu using shared memory.
 *
 * Stages per-block tiles of input data into shared memory before computing
 * the sin/cos/pow/relu expression, improving memory access locality.
 *
 * @param[in]  a  Pointer to the first input array (global memory).
 * @param[in]  b  Pointer to the second input array (global memory).
 * @param[out] c  Pointer to the output array (global memory).
 * @param[in]  N  Number of elements.
 */
__global__ void sin_cos_pow_relu_shared_kernel(const float* a, const float* b, float* c, int N)
{
    extern __shared__ float shmem[];
    float* s_a = shmem;
    float* s_b = shmem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    s_a[t] = (idx < N) ? a[idx] : 0.0f;
    s_b[t] = (idx < N) ? b[idx] : 0.0f;

    __syncthreads();

    if (idx < N)
    {
        float val = sinf(s_a[t]) + cosf(s_b[t]);
        val = powf(val, 2.0f);
        c[idx] = relu(val);
    }
}

/**
 * @brief CUDA kernel for vectorized sin_cos_pow_relu operation using float4.
 *
 * Each thread processes four elements packed in float4.
 * Applies sin, cos, pow, and ReLU operations to all elements.
 *
 * @param[in]  a  Pointer to the first input array (float4, global memory).
 * @param[in]  b  Pointer to the second input array (float4, global memory).
 * @param[out] c  Pointer to the output array (float4, global memory).
 * @param[in]  N  Number of float4 elements.
 */
__global__ void sin_cos_pow_relu_float4_kernel(const float4* a, const float4* b, float4* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float4 aval = a[idx];
        float4 bval = b[idx];
        float4 res;

        res.x = relu(powf(sinf(aval.x) + cosf(bval.x), 2.0f));
        res.y = relu(powf(sinf(aval.y) + cosf(bval.y), 2.0f));
        res.z = relu(powf(sinf(aval.z) + cosf(bval.z), 2.0f));
        res.w = relu(powf(sinf(aval.w) + cosf(bval.w), 2.0f));

        c[idx] = res;
    }
}