/**
 * @file    kernel_interface.hpp
 * @brief   Abstract interface for all kernel operations (CPU, GPU, etc.)
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Defines the abstract API for every math kernel in the benchmark suite.
 */

#pragma once

#include <string>
#include <vector>

/**
 * @class   KernelInterface
 * @brief   Base interface for all kernel operations.
 *
 * All kernels (CPU or GPU) must inherit from this interface.
 */
class KernelInterface
{
   public:
    virtual ~KernelInterface() = default;

    /**
     * @brief Get the name of the kernel operation.
     * @return Operation name (e.g. "add", "sin_cos_pow_relu")
     */
    virtual std::string name() const = 0;

    /**
     * @brief Run the kernel operation on the provided data.
     *
     * @param input1 Pointer to the first input buffer.
     * @param input2 Pointer to the second input buffer (can be nullptr for unary ops).
     * @param output Pointer to the output buffer.
     * @param size   Number of elements to process.
     */
    virtual void run(float* input1, float* input2, float* output, size_t size) = 0;
};