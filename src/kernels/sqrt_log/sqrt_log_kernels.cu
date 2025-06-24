#include "sqrt_log_kernels.cuh"
#include <math.h>

__global__ void sqrt_log_global_kernel(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = sqrtf(a[i]) + logf(b[i] + 1e-6f);
    }
}

__global__ void sqrt_log_shared_kernel(const float* a, const float* b, float* c, int N) {
    __shared__ float s_a[256];
    __shared__ float s_b[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    if (i < N) {
        s_a[t] = a[i];
        s_b[t] = b[i];
        __syncthreads();
        c[i] = sqrtf(s_a[t]) + logf(s_b[t] + 1e-6f);
    }
}

__global__ void sqrt_log_float4_kernel(const float4* a, const float4* b, float4* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 va = a[i];
        float4 vb = b[i];
        c[i] = make_float4(
            sqrtf(va.x) + logf(vb.x + 1e-6f),
            sqrtf(va.y) + logf(vb.y + 1e-6f),
            sqrtf(va.z) + logf(vb.z + 1e-6f),
            sqrtf(va.w) + logf(vb.w + 1e-6f)
        );
    }
}