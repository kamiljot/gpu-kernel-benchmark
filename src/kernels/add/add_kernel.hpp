/**
 * @file    add_kernel.hpp
 * @brief   Declaration for the AddKernel CUDA kernel class.
 * @author  Kamil J.
 * @date    2025-07-09
 */

#pragma once
#include <string>
#include <vector>

#include "kernel.hpp"

 /**
  * @class   AddKernel
  * @brief   Modular CUDA kernel launcher for element-wise addition.
  */
class AddKernel : public Kernel
{
public:
	AddKernel() = default;
	~AddKernel() override = default;

	std::string name() const override
	{
		return "add";
	}

	void launch(const std::vector<float>& input_a, const std::vector<float>& input_b,
		std::vector<float>& output) override;
};