/**
 * @file    profiler.hpp
 * @brief   Abstract profiler interface for kernel benchmarking and ML scenarios.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Declares the abstract Profiler interface for timing and collecting execution statistics.
 */

#pragma once

#include <string>

 /**
  * @class   Profiler
  * @brief   Abstract interface for benchmark profilers.
  */
class Profiler
{
public:
	virtual ~Profiler()
	{
	}

	/**
	 * @brief Starts profiling/timing.
	 */
	virtual void start() = 0;

	/**
	 * @brief Stops profiling/timing.
	 */
	virtual void stop() = 0;

	/**
	 * @brief Returns elapsed time in milliseconds.
	 */
	virtual double elapsed_time_ms() const = 0;

	/**
	 * @brief Saves profiling data to a JSON file.
	 * @param[in] filename Output JSON file name.
	 */
	virtual void save_to_json(const std::string& filename) const = 0;
};