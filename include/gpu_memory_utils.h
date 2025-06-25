#ifndef SIN_COS_POW_RELU_H
#define SIN_COS_POW_RELU_H

#ifdef __cplusplus
extern "C" {
#endif

	// Benchmark launcher functions for each kernel variant
	float run_sin_cos_pow_relu_global(const float* a, const float* b, float* c, int N);
	float run_sin_cos_pow_relu_shared(const float* a, const float* b, float* c, int N);
	float run_sin_cos_pow_relu_float4(const float* a, const float* b, float* c, int N);

	// CPU fallback implementation for reference
	float run_cpu_sin_cos_pow_relu(const float* a, const float* b, float* c, int N);

#ifdef __cplusplus
}
#endif

#endif // SIN_COS_POW_RELU_H
