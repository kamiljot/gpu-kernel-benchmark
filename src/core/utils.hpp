/**
 * @file    utils.hpp
 * @brief   Utility functions for modular CUDA/ML benchmark scenarios.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Declares utility functions for common operations such as random input generation and timing.
 */

#pragma once

#include <random>
#include <vector>

 /**
  * @brief Generates random float input arrays for benchmarking.
  *
  * @param[out] a  Output vector a, resized and filled with random floats.
  * @param[out] b  Output vector b, resized and filled with random floats.
  * @param[in]  N  Number of elements to generate.
  */
void generate_random_input(std::vector<float>& a, std::vector<float>& b, int N);

/**
 * @brief Returns the current time in milliseconds (high-resolution clock).
 */
double current_time_ms();