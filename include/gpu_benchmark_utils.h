#pragma once
#include <cuda_runtime.h>  

// Launches global memory kernel and measures execution time  
float benchmark_global_kernel(const float* d_a, const float* d_b, float* d_c, int N);  

// Launches shared memory kernel and measures execution time  
float benchmark_shared_kernel(const float* d_a, const float* d_b, float* d_c, int N, int blockSize);  

// Launches float4 kernel and measures execution time  
float benchmark_float4_kernel(const float* a, const float* b, int N);