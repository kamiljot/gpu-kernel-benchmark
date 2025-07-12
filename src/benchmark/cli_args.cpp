/**
 * @file    cli_args.cpp
 * @brief   CLI argument parsing for benchmark runner.
 * @author  Kamil J.
 * @date    2025-07-12
 */

#include "benchmark/cli_args.hpp"

#include <string>

Args parse_args(int argc, char* argv[])
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc)
            args.backend = argv[++i];
        else if (a == "--op" && i + 1 < argc)
            args.op = argv[++i];
        else if (a == "--variant" && i + 1 < argc)
            args.variant = argv[++i];
        else if (a == "--size" && i + 1 < argc)
            args.size = std::stoul(argv[++i]);
        else if (a == "--passes" && i + 1 < argc)
            args.passes = std::stoi(argv[++i]);
        else if (a == "--csv" && i + 1 < argc)
            args.csv_file = argv[++i];
    }
    return args;
}