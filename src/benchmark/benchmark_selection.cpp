/**
 * @file    benchmark_selection.cpp
 * @brief   Implementation for backend/kernel selection helpers.
 * @author  Kamil J.
 * @date    2025-07-12
 */

#include "benchmark/benchmark_selection.hpp"

#include <algorithm>

#include "backend/backend_registry.hpp"
#include "kernels/kernel_registry.hpp"

std::vector<std::string> select_backends(const std::string& backend_arg)
{
    if (backend_arg == "all")
        return BackendRegistry::instance().available_backends();
    else
        return {backend_arg};
}

std::vector<std::string> select_kernels(const std::string& op_arg, const std::string& variant_arg)
{
    std::vector<std::string> kernels;
    const auto& all = KernelRegistry::instance().available_kernels();
    if (op_arg == "all" && variant_arg == "all")
    {
        return all;
    }
    else if (variant_arg == "all")
    {
        for (const auto& name : all)
            if (name.find(op_arg + "_") == 0) kernels.push_back(name);
        return kernels;
    }
    else
    {
        std::string full = op_arg + "_" + variant_arg;
        if (std::find(all.begin(), all.end(), full) != all.end())
            return {full};
        else
            return {};
    }
}