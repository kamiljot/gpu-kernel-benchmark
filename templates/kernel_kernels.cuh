// Device kernel declarations for different {{name}} kernel variants.

#pragma once
#include <cuda_runtime.h>

// Global memory kernel: basic element-wise operation for {{name}}.
__global__ void{ {name} }_global_kernel(const float* a, const float* b, float* c, int N);

// Shared memory kernel: uses shared memory for improved performance.
__global__ void{ {name} }_shared_kernel(const float* a, const float* b, float* c, int N);

// float4 vectorized kernel: processes four elements at a time.
__global__ void{ {name} }_float4_kernel(const float4* a, const float4* b, float4* c, int N);