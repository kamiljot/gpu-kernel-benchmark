/**
 * @file    add_kernel.cpp
 * @brief   Implementation of AddGlobalKernel (CPU only).
 * @author  Kamil J.
 * @date    2025-07-10
 */

#include "kernels/add/add_kernel.hpp"

void AddGlobalKernel::run(float* in1, float* in2, float* out, size_t size)
{
    for (size_t i = 0; i < size; ++i) out[i] = in1[i] + in2[i];
}