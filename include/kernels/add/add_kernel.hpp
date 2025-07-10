/**
 * @file    add_kernel.hpp
 * @brief   KernelInterface wrapper for 'add_global' operation (CPU/GPU dispatch).
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Defines AddGlobalKernel, which performs vector addition on CPU or dispatches to GPU via backend.
 */

#pragma once

#include "kernels/kernel_interface.hpp"

 /**
  * @class   AddGlobalKernel
  * @brief   Modular kernel for vector addition (add_global).
  */
class AddGlobalKernel : public KernelInterface {
public:
	/**
	 * @brief   Constructs AddGlobalKernel.
	 */
	AddGlobalKernel() = default;

	/**
	 * @brief   Executes vector addition on CPU or dispatches to GPU backend.
	 * @param   in1    Pointer to first input array.
	 * @param   in2    Pointer to second input array.
	 * @param   out    Pointer to output array.
	 * @param   size   Number of elements.
	 */
	void run(float* in1, float* in2, float* out, size_t size) override;

	/**
	* @brief   Returns the kernel name ("add_global").
	* @return  Kernel name as a string.
	*/
	std::string name() const override { return "add_global"; }
};