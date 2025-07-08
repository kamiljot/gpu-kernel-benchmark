/**
 * @file    {{name}}_kernels.cu
 * @brief   CUDA kernel implementations for {{name}} kernels (global, shared, float4).
 * @author  Kamil J.
 * @date    {{date}}
 *
 * Implements CUDA device kernels for the '{{name}}' operation in all supported variants.
 */

#include "{{name}}_kernels.cuh"

/**
 * @brief Global memory kernel for '{{name}}' operation.
 */
__global__ void
{
    {
        name
    }
}
_global_kernel(const float* a, const float* b, float* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // TODO: Implement global kernel logic for {{name}}
        c[idx] = 0.0f;
    }
}

/**
 * @brief Shared memory kernel for '{{name}}' operation.
 */
__global__ void
{
    {
        name
    }
}
_shared_kernel(const float* a, const float* b, float* c, int N)
{
    extern __shared__ float shmem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // TODO: Implement shared kernel logic for {{name}} using shared memory
        c[idx] = 0.0f;
    }
}

/**
 * @brief float4 vectorized kernel for '{{name}}' operation.
 */
__global__ void
{
    {
        name
    }
}
_float4_kernel(const float4* a, const float4* b, float4* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // TODO: Implement float4 vectorized kernel logic for {{name}}
        c[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}