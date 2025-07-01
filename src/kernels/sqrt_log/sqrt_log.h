#pragma once

extern "C" float run_sqrt_log_global(const float* a, const float* b, float* c, int N);
extern "C" float run_sqrt_log_shared(const float* a, const float* b, float* c, int N);
extern "C" float run_sqrt_log_float4(const float* a, const float* b, float* c, int N);
