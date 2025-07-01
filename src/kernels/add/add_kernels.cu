// CUDA kernel implementations for the "add" operation: global memory, shared memory, and float4 variants.

#include "add_kernels.cuh"

__global__ void add_global_kernel(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_shared_kernel(const float* a, const float* b, float* c, int N) {
    __shared__ float s_a[256];
    __shared__ float s_b[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    if (i < N) {
        s_a[t] = a[i];
        s_b[t] = b[i];
        __syncthreads();
        c[i] = s_a[t] + s_b[t];
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