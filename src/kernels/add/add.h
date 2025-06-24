#pragma once
extern "C" float run_add_global(const float* a, const float* b, float* c, int N);
extern "C" float run_add_shared(const float* a, const float* b, float* c, int N);
extern "C" float run_add_float4(const float* a, const float* b, float* c, int N);