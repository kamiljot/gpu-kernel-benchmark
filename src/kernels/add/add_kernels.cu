// CUDA kernel implementations for the "add" operation: global memory, shared memory, and float4 variants.

#pragma once

#include "add_kernels.cuh"

__global__ void add_global_kernel(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

// Shared memory kernel: loads input into shared memory for improved memory access efficiency.
__global__ void add_shared_kernel(const float* a, const float* b, float* c, int N) {
    __shared__ float s_a[256];
    __shared__ float s_b[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load elements into shared memory, pad with zero if out of range
    s_a[tid] = (idx < N) ? a[idx] : 0.0f;
    s_b[tid] = (idx < N) ? b[idx] : 0.0f;

    __syncthreads(); // Synchronize to ensure all loads complete

    if (idx < N) {
        c[idx] = s_a[tid] + s_b[tid];
    }
}

__global__ void add_float4_kernel(const float4* a, const float4* b, float4* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 va = a[i];
        float4 vb = b[i];
        c[i] = make_float4(
            va.x + vb.x,
            va.y + vb.y,
            va.z + vb.z,
            va.w + vb.w
        );
    }
}