#include "benchmark_utils.h"
#include <random>

// Fill vectors a and b with random float values in range [0.1, 1.0]
void generate_input_data(std::vector<float>& a, std::vector<float>& b) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
}

// Write timing results to CSV output stream
void write_result(std::ofstream& out, int N, int pass,
    double cpu_time, float time_global,
    float time_shared, float time_float4) {
    out << N << "," << pass << "," << cpu_time << ","
        << time_global << "," << time_shared << "," << time_float4 << "\n";
}