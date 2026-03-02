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
 * @brief Generates N uniformly distributed random floats for two input vectors.
 *
 * The function resizes the output vectors to size N and fills them with pseudo-random
 * floating-point numbers in the range [0, 1). It is used to produce reproducible
 * inputs for benchmarks when no input file is provided.
 *
 * @param[in]  N  Number of elements to generate.
 * @param[out] a  Vector to be filled with random values for the first input.
 * @param[out] b  Vector to be filled with random values for the second input.
 */
void generate_random_input(int N, std::vector<float>& a, std::vector<float>& b);

/**
 * @brief Writes two float vectors to a binary file for later reuse.
 *
 * The format is simple: the function writes the size of the vector (int) followed by
 * raw float data for vector `a`, then similarly for vector `b`. This file is
 * compatible with the project's `read_input_file` helper.
 *
 * @param[in] filename  Path to the output file.
 * @param[in] a         Vector containing the first input.
 * @param[in] b         Vector containing the second input.
 */
void write_input_file(const std::string& filename, const std::vector<float>& a, const std::vector<float>& b);
