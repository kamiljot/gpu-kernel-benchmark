#include "gpu_benchmark_utils.h"
#include "gpu_memory_utils.h"

// Declare the CUDA kernel here (defined in gpu_kernel.cu)
__global__ void sqrt_log_kernel(const float*, const float*, float*, int);
__global__ void sqrt_log_shared_kernel(const float*, const float*, float*, int);
__global__ void sqrt_log_float4_kernel(const float4*, const float4*, float4*, int);

    // Measure execution time of the global memory kernel
    float benchmark_global_kernel(const float* d_a, const float* d_b, float* d_c, int N) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        cudaEventRecord(start);
        sqrt_log_kernel << <numBlocks, blockSize >> > (d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms;
    }

    // Measure execution time of the shared memory kernel
    float benchmark_shared_kernel(const float* d_a, const float* d_b, float* d_c, int N, int blockSize) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        int numBlocks = (N + blockSize - 1) / blockSize;
        size_t sharedSize = 2 * blockSize * sizeof(float);
        cudaEventRecord(start);
        sqrt_log_shared_kernel << <numBlocks, blockSize, sharedSize >> > (d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms;
    }

    // Measure execution time of the float4 kernel
    float benchmark_float4_kernel(const float* a, const float* b, int N) {
        int N_vec4 = N / 4;
        float4* d_a4, * d_b4, * d_c4;
        allocate_and_copy_vec4(a, b, &d_a4, &d_b4, &d_c4, N_vec4);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        int blockSize = 256;
        int numBlocks = (N_vec4 + blockSize - 1) / blockSize;
        cudaEventRecord(start);
        sqrt_log_float4_kernel << <numBlocks, blockSize >> > (d_a4, d_b4, d_c4, N_vec4);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        cudaFree(d_a4);
        cudaFree(d_b4);
        cudaFree(d_c4);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return ms;
    }