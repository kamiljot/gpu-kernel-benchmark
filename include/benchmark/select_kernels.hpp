/**
 * @file    select_kernels.hpp
 * @brief   Kernel selection helper for benchmarking.
 * @author  Kamil J.
 * @date    2025-07-12
 */
#pragma once
#include <string>
#include <vector>

/**
 * @brief Selects kernel variants based on op and variant.
 * @param all_kernels   List of all registered kernels.
 * @param op            Kernel operation (e.g., "add").
 * @param variant       Variant name (e.g., "global" or "all").
 * @return              Vector of selected kernel names.
 */
std::vector<std::string> select_kernels(const std::vector<std::string>& all_kernels, const std::string& op,
                                        const std::string& variant);