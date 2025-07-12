/**
 * @file    generators.cpp
 * @brief   Input data generators for benchmarking.
 * @author  Kamil J.
 * @date    2025-07-12
 */

#include "benchmark/generators.hpp"

#include <random>

std::vector<float> generate_input(size_t size)
{
    std::vector<float> data(size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) v = dist(rng);
    return data;
}