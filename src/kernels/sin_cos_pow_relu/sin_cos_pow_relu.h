#pragma once

#ifdef __cplusplus
extern "C" {
#endif

	float run_sin_cos_pow_relu_global(const float* a, const float* b, float* c, int N);
	float run_sin_cos_pow_relu_shared(const float* a, const float* b, float* c, int N);
	float run_sin_cos_pow_relu_float4(const float* a, const float* b, float* c, int N);
	float run_cpu_sin_cos_pow_relu(const float* a, const float* b, float* c, int N);

#ifdef __cplusplus
}
#endif