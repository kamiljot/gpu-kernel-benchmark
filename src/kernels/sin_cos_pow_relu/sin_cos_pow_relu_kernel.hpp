/**
 * @file    sin_cos_pow_relu_kernel.hpp
 * @brief   Declaration for the SinCosPowReluKernel CUDA kernel class.
 * @author  Kamil J.
 * @date    2025-07-09
 */

#pragma once
#include <string>
#include <vector>

#include "kernel.hpp"

 /**
  * @class   SinCosPowReluKernel
  * @brief   Modular CUDA kernel launcher for sin, cos, pow, and ReLU.
  */
class SinCosPowReluKernel : public Kernel
{
public:
	SinCosPowReluKernel() = default;
	~SinCosPowReluKernel() override = default;
	std::string name() const override
	{
		return "sin_cos_pow_relu";
	}
	void launch(const std::vector<float>& input_a, const std::vector<float>& input_b,
		std::vector<float>& output) override;
};