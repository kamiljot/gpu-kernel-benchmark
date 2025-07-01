// Input/output utilities for benchmark runs and CSV result logging.

#pragma once
#include <vector>
#include <string>

// Loads input vectors a and b from a binary file
bool read_input_file(const std::string& filename,
    std::vector<float>& a,
    std::vector<float>& b);

// Appends one benchmark result to CSV output file
void append_result_to_csv(const std::string& filename,
    const std::string& operation,
    int N,
    const struct BenchmarkResult& result);