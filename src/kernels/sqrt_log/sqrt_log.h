// Host launchers for sqrt_log kernel variants (global, shared memory, float4).

#pragma once

// Runs the global memory sqrt_log kernel.
// N: number of elements.
extern "C" float run_sqrt_log_global(const float* a, const float* b, float* c, int N);

// Runs the shared memory sqrt_log kernel.
// N: number of elements.
extern "C" float run_sqrt_log_shared(const float* a, const float* b, float* c, int N);

// Runs the float4 vectorized sqrt_log kernel.
// N: number of elements.
extern "C" float run_sqrt_log_float4(const float* a, const float* b, float* c, int N);