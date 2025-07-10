/**
 * @file    add_kernel.cpp
 * @brief   Implements the AddKernel class.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Adds input1 and input2 elementwise and writes result to output.
 */

#include "add_kernel.hpp"

void AddKernel::run(float* input1, float* input2, float* output, size_t size) {
	for (size_t i = 0; i < size; ++i)
		output[i] = input1[i] + input2[i];
}