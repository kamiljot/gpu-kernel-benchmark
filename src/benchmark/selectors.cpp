/**
 * @file    selectors.cpp
 * @brief   Helper functions for backend/kernel selection for benchmarking CLI (implementation).
 * @author  Kamil J.
 * @date    2025-07-12
 */

#include "benchmark/selectors.hpp"

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
    std::vector<std::string> all_kernels = KernelRegistry::instance().available_kernels();
    std::vector<std::string> selected;

    if (op_arg == "all" && variant_arg == "all") return all_kernels;

    if (op_arg == "all")
    {
        // All ops, single variant
        for (const auto& k : all_kernels)
            if (k.size() > variant_arg.size() && k.substr(k.size() - variant_arg.size()) == variant_arg)
                selected.push_back(k);
        return selected;
    }
    if (variant_arg == "all")
    {
        // Single op, all variants
        for (const auto& k : all_kernels)
            if (k.find(op_arg) == 0) selected.push_back(k);
        return selected;
    }
    // Single op, single variant
    std::string name = op_arg + "_" + variant_arg;
    if (std::find(all_kernels.begin(), all_kernels.end(), name) != all_kernels.end()) return {name};
    return {};
}