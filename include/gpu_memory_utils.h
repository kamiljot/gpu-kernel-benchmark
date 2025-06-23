#pragma once
#include <cuda_runtime.h>

// Allocates GPU memory for input arrays a, b and output array c,
// and copies input data from host to device.
void allocate_and_copy(const float* a, const float* b,
    float** d_a, float** d_b, float** d_c, int N);

// Allocates GPU memory for float4 input arrays a, b and output array c,
// and copies input data (already packed into float4) from host to device.
void allocate_and_copy_vec4(const float* a, const float* b,
    float4** d_a4, float4** d_b4, float4** d_c4, int N_vec4);

// Frees memory for all float* device arrays.
void free_device(float* d_a, float* d_b, float* d_c);