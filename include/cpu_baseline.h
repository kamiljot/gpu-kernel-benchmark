#pragma once
#include <string>

float run_cpu_baseline(const std::string& operation, const float* a, const float* b, float* c, int N);
float run_cpu_add(const float* a, const float* b, float* c, int N);
float run_cpu_sqrt_log(const float* a, const float* b, float* c, int N);