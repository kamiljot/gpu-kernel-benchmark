#pragma once
#include <cuda_runtime.h>
#include <tuple>
#include <stdexcept> 
#include <cstdio>
#include <vector>
#include <algorithm>

// Macro for checking CUDA errors in host code.
#define CHECK_CUDA(call)                                                      \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            throw std::runtime_error(cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

// Persistent buffer structure for managing device memory and data transfer
struct PersistentBuffer {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    int N = 0;
    bool initialized = false;

    void allocate(int size) {
        if (initialized && size == N) return;

        if (initialized) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }

        CHECK_CUDA(cudaMalloc(&d_a, size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b, size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_c, size * sizeof(float)));

        N = size;
        initialized = true;
    }

    void copy_to_device(const float* a, const float* b, int size) {
        if (!initialized || size != N)
            allocate(size);
        CHECK_CUDA(cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice));
    }

    void copy_to_host(float* c) const {
        CHECK_CUDA(cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void free_buffers() {
        if (initialized) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            initialized = false;
        }
    }

    ~PersistentBuffer() {
        free_buffers();
    }
};

// Measures time taken to launch and complete a kernel using CUDA events
template<typename KernelFunc>
float measure_kernel_time(KernelFunc kernel_call) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    kernel_call();

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return time_ms;
}

// Allocates device memory and copies host arrays a, b to device
inline std::tuple<float*, float*, float*> allocate_and_copy_to_device(const float* a, const float* b, int N) {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    size_t size = N * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));
    CHECK_CUDA(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
    return { d_a, d_b, d_c };
}

// Copies device array d_c to host array c and frees all device memory
inline void copy_from_device_and_free(float* c, float* d_c, float* d_a, float* d_b, int N) {
    size_t size = N * sizeof(float);
    CHECK_CUDA(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// float4 variants

inline std::tuple<float4*, float4*, float4*> allocate_and_copy_to_device_float4(const float4* a, const float4* b, int N_vec4) {
    float4* d_a = nullptr;
    float4* d_b = nullptr;
    float4* d_c = nullptr;
    size_t size = N_vec4 * sizeof(float4);
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));
    CHECK_CUDA(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
    return { d_a, d_b, d_c };
}

inline void copy_from_device_and_free_float4(float* c, float4* d_c, float4* d_a, float4* d_b, int N_vec4) {
    size_t size = N_vec4 * sizeof(float4);
    CHECK_CUDA(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

// Measures execution time of a kernel by launching it multiple times and averaging.
template <typename KernelFunc>
float launch_kernel_multiple_times(KernelFunc kernel, int passes) {
    cudaEvent_t start, stop;
    float total_ms = -1.0f;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < passes; ++i) {
        kernel();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return total_ms / passes;
}

// Packs input float array into padded float4 vector suitable for device copy.
inline std::vector<float4> pack_and_pad_to_float4(const float* data, int N) {
    int padded_N = (N + 3) / 4 * 4;
    std::vector<float> padded_data(padded_N, 0.0f);
    std::copy(data, data + N, padded_data.begin());

    int N_vec4 = padded_N / 4;
    std::vector<float4> packed(N_vec4);

    for (int i = 0; i < N_vec4; ++i) {
        packed[i] = make_float4(
            padded_data[4 * i],
            padded_data[4 * i + 1],
            padded_data[4 * i + 2],
            padded_data[4 * i + 3]
        );
    }
    return packed;
}