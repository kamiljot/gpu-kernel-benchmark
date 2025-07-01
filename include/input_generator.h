// Tools for random input generation for benchmark data.

#pragma once
#include <vector>
#include <string>

// Generates N random float inputs for vectors a and b
void generate_random_input(int N, std::vector<float>& a, std::vector<float>& b);

// Writes vectors a and b to a binary file
void write_input_file(const std::string& filename, const std::vector<float>& a, const std::vector<float>& b);