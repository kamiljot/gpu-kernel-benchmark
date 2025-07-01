// Input/output utilities for reading binary input and saving results to CSV.

#include "benchmark_utils.h"
#include "kernel_dispatch.h"
#include "input_generator.h"
#include <iostream>
#include <fstream>

bool read_input_file(const std::string& filename,
    std::vector<float>& a,
    std::vector<float>& b) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        // Auto-generate input if file doesn't exist
        std::cout << "Input file not found. Generating random data...\n";
        int N = 1000000;
        generate_random_input(N, a, b);
        write_input_file(filename, a, b);
        return true;
    }

    int N;
    file.read(reinterpret_cast<char*>(&N), sizeof(int));
    a.resize(N);
    b.resize(N);
    file.read(reinterpret_cast<char*>(a.data()), N * sizeof(float));
    file.read(reinterpret_cast<char*>(b.data()), N * sizeof(float));
    return file.good();
}

void append_result_to_csv(const std::string& filename,
    const std::string& operation,
    int N,
    const BenchmarkResult& result) {
    std::ofstream file(filename, std::ios::app);
    file << operation << "," << N << ","
        << result.cpu_time << ","
        << result.gpu_global_time << ","
        << result.gpu_shared_time << ","
        << result.gpu_float4_time << "\n";
}