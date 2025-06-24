#pragma once

// Public C-compatible API for sqrt_log kernel variants
extern "C" float run_sqrt_log_global(const float* a, const float* b, float* c, int N);
extern "C" float run_sqrt_log_shared(const float* a, const float* b, float* c, int N);
extern "C" float run_sqrt_log_float4(const float* a, const float* b, float* c, int N);
