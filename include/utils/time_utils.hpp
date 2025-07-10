/**
 * @file    time_utils.hpp
 * @brief   Utilities for measuring elapsed time on CPU/GPU (ready for future CUDA).
 * @author  Kamil J.
 * @date    2025-07-10
 */

#pragma once

#include <chrono>

 /**
  * @class   CpuTimer
  * @brief   Simple wall-clock timer for CPU benchmarks.
  */
class CpuTimer {
public:
	void start() { start_ = std::chrono::high_resolution_clock::now(); }
	void stop() { end_ = std::chrono::high_resolution_clock::now(); }

	/// Returns elapsed time in milliseconds
	double elapsed_ms() const {
		return std::chrono::duration<double, std::milli>(end_ - start_).count();
	}

private:
	std::chrono::high_resolution_clock::time_point start_, end_;
};