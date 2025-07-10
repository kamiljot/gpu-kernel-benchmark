/**
 * @file    utils.cpp
 * @brief   Utility function implementations for modular CUDA/ML benchmark scenarios.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Implements utility functions such as random input generation and time measurement.
 */

#include "utils.hpp"

#include <chrono>

void generate_random_input(std::vector<float>& a, std::vector<float>& b, int N)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	a.resize(N);
	b.resize(N);
	for (int i = 0; i < N; ++i)
	{
		a[i] = dist(gen);
		b[i] = dist(gen);
	}
}

double current_time_ms()
{
	using namespace std::chrono;
	return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count() / 1000.0;
}