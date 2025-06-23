#include <cuda_runtime.h>
#include <math.h>

// Global memory CUDA kernel: compute sqrt(a[i]) + log(b[i])
__global__ void sqrt_log_kernel(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = sqrtf(a[i]) + logf(b[i] + 1e-6f);
    }
}

// Shared memory kernel: loads input to shared memory before computing
__global__ void sqrt_log_shared_kernel(const float* a, const float* b, float* c, int N) {
    extern __shared__ float shared[];
    float* sh_a = shared;
    float* sh_b = shared + blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        sh_a[threadIdx.x] = a[i];
        sh_b[threadIdx.x] = b[i];
        __syncthreads();
        c[i] = sqrtf(sh_a[threadIdx.x]) + logf(sh_b[threadIdx.x] + 1e-6f);
    }
}

// Vectorized kernel with float4 memory access
__global__ void sqrt_log_float4_kernel(const float4* a, const float4* b, float4* c, int N_vec4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_vec4) {
        float4 av = a[i];
        float4 bv = b[i];
        float4 cv;
        cv.x = sqrtf(av.x) + logf(bv.x + 1e-6f);
        cv.y = sqrtf(av.y) + logf(bv.y + 1e-6f);
        cv.z = sqrtf(av.z) + logf(bv.z + 1e-6f);
        cv.w = sqrtf(av.w) + logf(bv.w + 1e-6f);
        c[i] = cv;
    }
}