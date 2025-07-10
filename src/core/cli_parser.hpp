/**
 * @file    cli_parser.hpp
 * @brief   Simple CLI parser for modular CUDA/ML benchmark scenarios.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Declares a minimal command-line argument parser for selecting scenario, kernel, and backend.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

 /**
  * @class   CliParser
  * @brief   Parses CLI arguments for the benchmark application.
  */
class CliParser
{
public:
	CliParser(int argc, char** argv);

	/**
	 * @brief Retrieves the value for a CLI option or a default value.
	 */
	std::string get_option(const std::string& option, const std::string& default_value = "") const;

	/**
	 * @brief Checks if a flag (boolean CLI switch) is present.
	 */
	bool has_flag(const std::string& flag) const;

private:
	std::unordered_map<std::string, std::string> options_;
	std::vector<std::string> flags_;
};