/**
 * @file    selectors.hpp
 * @brief   Helper functions for backend/kernel selection for benchmarking CLI.
 * @author  Kamil J.
 * @date    2025-07-12
 */
#pragma once

#include <string>
#include <vector>

/**
 * @brief   Selects backend(s) for benchmarking based on CLI argument.
 * @param   backend_arg   Argument string, e.g. "cpu", "cuda" or "all"
 * @return  Vector of backend names to use.
 */
std::vector<std::string> select_backends(const std::string& backend_arg);

/**
 * @brief   Selects kernel(s) for benchmarking based on op/variant CLI arguments.
 * @param   op_arg       Operation name, e.g. "add" or "all"
 * @param   variant_arg  Variant name, e.g. "global" or "all"
 * @return  Vector of kernel names to use.
 */
std::vector<std::string> select_kernels(const std::string& op_arg, const std::string& variant_arg);