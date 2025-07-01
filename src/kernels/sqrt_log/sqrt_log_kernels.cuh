// Device kernel declarations for different sqrt_log kernel implementations.
#pragma once
#include <cuda_runtime.h>

// Global memory kernel: computes sqrt(log(a) + log(b)) element-wise.
__global__ void sqrt_log_global_kernel(const float* a, const float* b, float* c, int N);

// Shared memory kernel: uses shared memory for faster log/sqrt computation.
__global__ void sqrt_log_shared_kernel(const float* a, const float* b, float* c, int N);

// float4 vectorized kernel: operates on four elements at a time.
__global__ void sqrt_log_float4_kernel(const float4* a, const float4* b, float4* c, int N);