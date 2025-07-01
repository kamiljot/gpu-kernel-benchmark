// CUDA kernel implementations for sin_cos_pow_relu (global, shared, float4).

#include "sin_cos_pow_relu_kernels.cuh"
#include <cmath>

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__global__ void sin_cos_pow_relu_global_kernel(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = sinf(a[idx]) + cosf(b[idx]);
        val = powf(val, 2.0f);
        c[idx] = relu(val);
    }
}

__global__ void sin_cos_pow_relu_shared_kernel(const float* a, const float* b, float* c, int N) {
    extern __shared__ float shmem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Here could be a shared memory optimized version; simple fallback:
        float val = sinf(a[idx]) + cosf(b[idx]);
        val = powf(val, 2.0f);
        c[idx] = relu(val);
    }
}

__global__ void sin_cos_pow_relu_float4_kernel(const float4* a, const float4* b, float4* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
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
