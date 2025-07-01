// Random input generation and binary file I/O for benchmarks.

#include "input_generator.h"
#include <fstream>
#include <random>
#include <iostream>

void generate_random_input(int N, std::vector<float>& a, std::vector<float>& b) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.01f, 10.0f);
    a.resize(N);
    b.resize(N);
    for (int i = 0; i < N; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    std::cout << "Generated N = " << N << "\n";
}

void write_input_file(const std::string& filename, const std::vector<float>& a, const std::vector<float>& b) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    int N = static_cast<int>(a.size());
    file.write(reinterpret_cast<const char*>(&N), sizeof(int));
    file.write(reinterpret_cast<const char*>(a.data()), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(b.data()), N * sizeof(float));
}