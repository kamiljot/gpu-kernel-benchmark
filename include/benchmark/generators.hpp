/**
 * @file    generators.hpp
 * @brief   Input data generators for benchmarking.
 * @author  Kamil J.
 * @date    2025-07-12
 */

#pragma once
#include <vector>

/**
 * @brief Generates a vector of random float input data.
 * @param size Number of elements.
 * @return Vector with random data.
 */
std::vector<float> generate_input(size_t size);