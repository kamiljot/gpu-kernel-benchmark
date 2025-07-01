// Device kernel declarations for different add kernel implementations.

#pragma once
#include <cuda_runtime.h>

// Global memory kernel: basic element-wise addition.
__global__ void add_global_kernel(const float* a, const float* b, float* c, int N);

// Shared memory kernel: uses shared memory for block-level efficiency.
__global__ void add_shared_kernel(const float* a, const float* b, float* c, int N);

// float4 vectorized kernel: processes four elements at a time using float4.
__global__ void add_float4_kernel(const float4* a, const float4* b, float4* c, int N);