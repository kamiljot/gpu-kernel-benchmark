/**
 * @file    config.hpp
 * @brief   Configuration loader for modular CUDA/ML benchmark scenarios.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Declares a simple configuration loader and data structure for benchmark settings.
 */

#pragma once

#include <string>
#include <unordered_map>

 /**
  * @class   Config
  * @brief   Holds configuration options for the benchmark.
  */
class Config
{
public:
	/**
	 * @brief Loads configuration from a key=value file.
	 */
	bool load_from_file(const std::string& filename);

	/**
	 * @brief Retrieves a configuration value as a string.
	 */
	std::string get(const std::string& key, const std::string& default_value = "") const;

	/**
	 * @brief Retrieves a configuration value as an int.
	 */
	int get_int(const std::string& key, int default_value = 0) const;

private:
	std::unordered_map<std::string, std::string> options_;
};