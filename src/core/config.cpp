/**
 * @file    config.cpp
 * @brief   Configuration loader implementation for modular CUDA/ML benchmark scenarios.
 * @author  Kamil J.
 * @date    2025-07-09
 *
 * Implements a simple loader for benchmark configuration options from a file.
 */

#include "config.hpp"

#include <fstream>
#include <sstream>

bool Config::load_from_file(const std::string& filename)
{
	std::ifstream infile(filename);
	if (!infile.is_open()) return false;

	std::string line;
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		std::string key, value;
		if (std::getline(iss, key, '=') && std::getline(iss, value))
		{
			options_[key] = value;
		}
	}
	return true;
}

std::string Config::get(const std::string& key, const std::string& default_value) const
{
	auto it = options_.find(key);
	if (it != options_.end()) return it->second;
	return default_value;
}

int Config::get_int(const std::string& key, int default_value) const
{
	auto it = options_.find(key);
	if (it != options_.end())
	{
		try
		{
			return std::stoi(it->second);
		}
		catch (...)
		{
		}
	}
	return default_value;
}