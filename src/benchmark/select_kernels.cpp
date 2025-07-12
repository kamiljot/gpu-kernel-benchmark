/**
 * @file    select_kernels.cpp
 * @brief   Kernel selection helper for benchmarking.
 * @author  Kamil J.
 * @date    2025-07-12
 */
#include "benchmark/select_kernels.hpp"

#include <algorithm>

std::vector<std::string> select_kernels(const std::vector<std::string>& all_kernels, const std::string& op,
                                        const std::string& variant)
{
    std::vector<std::string> selected;
    if (variant == "all")
    {
        for (const auto& name : all_kernels)
        {
            if (name.find(op) == 0) selected.push_back(name);
        }
    }
    else
    {
        std::string full_name = op + "_" + variant;
        if (std::find(all_kernels.begin(), all_kernels.end(), full_name) != all_kernels.end())
            selected.push_back(full_name);
    }
    return selected;
}