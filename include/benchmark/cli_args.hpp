/**
 * @file    cli_args.hpp
 * @brief   CLI argument parsing for benchmark runner.
 * @author  Kamil J.
 * @date    2025-07-12
 */

#pragma once
#include <string>

/**
 * @struct Args
 * @brief Structure holding parsed CLI arguments.
 */
struct Args
{
    std::string backend = "cpu";
    std::string op = "add";
    std::string variant = "all";
    size_t size = 1024 * 1024;
    int passes = 10;
    std::string csv_file = "results.csv";
};

/**
 * @brief Parses CLI arguments.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Parsed Args struct.
 */
Args parse_args(int argc, char* argv[]);