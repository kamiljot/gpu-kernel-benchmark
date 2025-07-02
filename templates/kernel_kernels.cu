// CUDA kernel implementations for {{name}} kernels.

#include "{{name}}_kernels.cuh"

__global__ void{ {name} }_global_kernel(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO: Implement global kernel logic for {{name}}
        c[idx] = 0.0f;
    }
}

__global__ void{ {name} }_shared_kernel(const float* a, const float* b, float* c, int N) {
    extern __shared__ float shmem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO: Implement shared kernel logic for {{name}} using shared memory
        c[idx] = 0.0f;
    }
}

__global__ void{ {name} }_float4_kernel(const float4* a, const float4* b, float4* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // TODO: Implement float4 vectorized kernel logic for {{name}}
        c[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}