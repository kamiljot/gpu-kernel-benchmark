#pragma once
#include <chrono>

// Simple utility class for measuring elapsed time in milliseconds
class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start_;
        return duration.count();
    }
};
