/**
 * @file    input_generator.h
 * @brief   Tools for random input generation for benchmark data.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Provides utility functions for generating and saving random input vectors for benchmarking.
 */

#pragma once
#include <string>
#include <vector>

/**
 * @brief Generates N random float inputs for vectors a and b.
 *
 * @param[in]  N  Number of elements to generate.
 * @param[out] a  Vector to be filled with random values for the first input.
 * @param[out] b  Vector to be filled with random values for the second input.
 */
void generate_random_input(int N, std::vector<float>& a, std::vector<float>& b);

/**
 * @brief Writes vectors a and b to a binary file.
 *
 * @param[in] filename  Path to the output file.
 * @param[in] a         Vector containing the first input.
 * @param[in] b         Vector containing the second input.
 */
void write_input_file(const std::string& filename, const std::vector<float>& a, const std::vector<float>& b);