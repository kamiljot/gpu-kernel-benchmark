// Host launchers for sin_cos_pow_relu kernel variants (global, shared memory, float4).

#pragma once

// Runs the global memory sin_cos_pow_relu kernel.
// N: number of elements.
extern "C" float run_sin_cos_pow_relu_global(const float* a, const float* b, float* c, int N);

// Runs the shared memory sin_cos_pow_relu kernel.
// N: number of elements.
extern "C" float run_sin_cos_pow_relu_shared(const float* a, const float* b, float* c, int N);

// Runs the float4 vectorized sin_cos_pow_relu kernel.
// N: number of elements.
extern "C" float run_sin_cos_pow_relu_float4(const float* a, const float* b, float* c, int N);
