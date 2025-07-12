/**
 * @file    add_kernel.hpp
 * @brief   Declares the AddKernel implementation (CPU).
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Example implementation of a CPU add kernel.
 */

#pragma once

#include "kernels/kernel_interface.hpp"

/**
 * @class   AddKernel
 * @brief   Example implementation of the Add kernel (CPU).
 */
class AddKernel : public KernelInterface
{
   public:
    /**
     * @brief Get the name of the kernel ("add").
     * @return Name of the operation.
     */
    std::string name() const override
    {
        return "add";
    }

    /**
     * @brief Run the add operation on CPU.
     *
     * @param input1 Pointer to the first input buffer.
     * @param input2 Pointer to the second input buffer.
     * @param output Pointer to the output buffer.
     * @param size   Number of elements to process.
     */
    void run(float* input1, float* input2, float* output, size_t size) override;
};