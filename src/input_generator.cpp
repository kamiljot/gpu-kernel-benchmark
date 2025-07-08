/**
 * @file    input_generator.cpp
 * @brief   Random input generation and binary file I/O for benchmarks.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Implements functions for generating random input data and saving/loading it from binary files.
 */

#include "input_generator.h"

#include <fstream>
#include <iostream>
#include <random>

/**
 * @brief Generates N random float inputs for vectors a and b.
 *
 * Fills both vectors with uniformly distributed random values in [0.01, 10.0].
 *
 * @param[in]  N  Number of elements to generate.
 * @param[out] a  Vector to be filled with random values for the first input.
 * @param[out] b  Vector to be filled with random values for the second input.
 */
void generate_random_input(int N, std::vector<float>& a, std::vector<float>& b)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.01f, 10.0f);
    a.resize(N);
    b.resize(N);
    for (int i = 0; i < N; ++i)
    {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    std::cout << "Generated N = " << N << "\n";
}

/**
 * @brief Writes vectors a and b to a binary file.
 *
 * The output format is: [int N][a array][b array], where N is the number of elements.
 *
 * @param[in] filename  Path to the output file.
 * @param[in] a         Vector containing the first input.
 * @param[in] b         Vector containing the second input.
 * @throws std::runtime_error if the file cannot be opened.
 */
void write_input_file(const std::string& filename, const std::vector<float>& a, const std::vector<float>& b)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    int N = static_cast<int>(a.size());
    file.write(reinterpret_cast<const char*>(&N), sizeof(int));
    file.write(reinterpret_cast<const char*>(a.data()), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(b.data()), N * sizeof(float));
}