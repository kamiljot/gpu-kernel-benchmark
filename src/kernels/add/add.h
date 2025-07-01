// Host launchers for different add kernel variants (global, shared memory, float4).

#pragma once

// Runs the global memory add kernel.
// N: number of elements.
extern "C" float run_add_global(const float* a, const float* b, float* c, int N);

// Runs the shared memory add kernel.
// N: number of elements.
extern "C" float run_add_shared(const float* a, const float* b, float* c, int N);

// Runs the float4 vectorized add kernel.
// N: number of elements.
extern "C" float run_add_float4(const float* a, const float* b, float* c, int N);