/**
 * @file    benchmark_selection.hpp
 * @brief   Helpers for selecting backends and kernels from CLI arguments.
 * @author  Kamil J.
 * @date    2025-07-12
 */

#pragma once
#include <string>
#include <vector>

/**
 * @brief Select backends based on CLI argument.
 * @param backend_arg User CLI argument (may be "all").
 * @return Vector of backend names.
 */
std::vector<std::string> select_backends(const std::string& backend_arg);

/**
 * @brief Select kernel names based on CLI args.
 * @param op_arg      Operation name or "all".
 * @param variant_arg Variant name or "all".
 * @return Vector of kernel names.
 */
std::vector<std::string> select_kernels(const std::string& op_arg, const std::string& variant_arg);