// Host launchers for different {{name}} kernel variants (global, shared memory, float4).

#pragma once

// Runs the global memory {{name}} kernel.
// N: number of elements.
extern "C" float run_{ {name} }_global(const float* a, const float* b, float* c, int N);

// Runs the shared memory {{name}} kernel.
// N: number of elements.
extern "C" float run_{ {name} }_shared(const float* a, const float* b, float* c, int N);

// Runs the float4 vectorized {{name}} kernel.
// N: number of elements.
extern "C" float run_{ {name} }_float4(const float* a, const float* b, float* c, int N);