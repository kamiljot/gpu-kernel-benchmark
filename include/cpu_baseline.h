// Reference CPU implementations for benchmarking purposes.

#pragma once

float run_cpu_add(const float* a, const float* b, float* c, int N);
float run_cpu_sqrt_log(const float* a, const float* b, float* c, int N);
float run_cpu_sin_cos_pow_relu(const float* a, const float* b, float* c, int N);
