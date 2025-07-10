/**
 * @file    cli_parser.cpp
 * @brief   Simple CLI parser implementation for modular CUDA/ML benchmark scenarios.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Implements a minimal command-line argument parser for scenario, kernel, and backend selection.
 */

#include "cli_parser.hpp"

CliParser::CliParser(int argc, char** argv)
{
	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-')
		{
			std::string opt = arg.substr(2);
			if ((i + 1) < argc && argv[i + 1][0] != '-')
			{
				options_[opt] = argv[i + 1];
				++i;
			}
			else
			{
				flags_.push_back(opt);
			}
		}
	}
}

std::string CliParser::get_option(const std::string& option, const std::string& default_value) const
{
	auto it = options_.find(option);
	if (it != options_.end()) return it->second;
	return default_value;
}

bool CliParser::has_flag(const std::string& flag) const
{
	return std::find(flags_.begin(), flags_.end(), flag) != flags_.end();
}