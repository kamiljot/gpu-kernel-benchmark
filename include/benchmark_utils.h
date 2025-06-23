#pragma once
#include <vector>
#include <fstream>

// Generates random float values for vectors a and b
void generate_input_data(std::vector<float>& a, std::vector<float>& b);

// Writes a single result line to the output CSV file
void write_result(std::ofstream& out, int N, int pass,
    double cpu_time, float time_global,
    float time_shared, float time_float4);
