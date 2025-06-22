#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for element-wise computation: c[i] = sqrt(a[i]) + log(b[i])
__global__ void sqrt_log_kernel(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = sqrtf(a[i]) + logf(b[i] + 1e-6f);
    }
}

// Host function that manages GPU memory and timing, and launches the kernel
extern "C" {
    void gpu_math(const float* a, const float* b, float* c, int N, float* kernel_time_ms) {
        float* d_a, * d_b, * d_c;
        size_t size = N * sizeof(float);

        // Allocate memory on device
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);

        // Copy data from host to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;

        // Measure kernel execution time using CUDA events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        sqrt_log_kernel << <numBlocks, blockSize >> > (d_a, d_b, d_c, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(kernel_time_ms, start, stop);

        // Copy results back to host
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}