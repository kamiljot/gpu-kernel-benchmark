#pragma once
#include <vector>
#include <string>

// Generates N random float inputs for vectors a and b
void generate_random_input(int N, std::vector<float>& a, std::vector<float>& b);

// Writes vectors a and b to a binary file in the expected format
void write_input_file(const std::string& filename, const std::vector<float>& a, const std::vector<float>& b);