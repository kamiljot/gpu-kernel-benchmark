#include "sin_cos_pow_relu_kernels.cuh"
#include <math.h>

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__global__ void sin_cos_pow_relu_global_kernel(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = relu(powf(sinf(a[i]) + cosf(b[i]), 2.0f));
    }
}

__global__ void sin_cos_pow_relu_shared_kernel(const float* a, const float* b, float* c, int N) {
    extern __shared__ float shared[];
    float* sa = shared;
    float* sb = &shared[blockDim.x];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        sa[threadIdx.x] = a[i];
        sb[threadIdx.x] = b[i];
        __syncthreads();
        c[i] = relu(powf(sinf(sa[threadIdx.x]) + cosf(sb[threadIdx.x]), 2.0f));
    }
}

__global__ void sin_cos_pow_relu_float4_kernel(const float4* a, const float4* b, float4* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int index = i * 4;
    if (index + 3 < N) {
        float4 va = reinterpret_cast<const float4*>(a)[i];
        float4 vb = reinterpret_cast<const float4*>(b)[i];
        float4 r;
        r.x = relu(powf(sinf(va.x) + cosf(vb.x), 2.0f));
        r.y = relu(powf(sinf(va.y) + cosf(vb.y), 2.0f));
        r.z = relu(powf(sinf(va.z) + cosf(vb.z), 2.0f));
        r.w = relu(powf(sinf(va.w) + cosf(vb.w), 2.0f));
        reinterpret_cast<float4*>(c)[i] = r;
    }
}