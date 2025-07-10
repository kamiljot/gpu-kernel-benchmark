/**
 * @file    kernel.hpp
 * @brief   Abstract kernel interface for modular GPU benchmarking.
 * @author  Kamil J.
 * @date    2025-07-09
 */

#pragma once
#include <string>
#include <vector>

 /**
  * @class   Kernel
  * @brief   Abstract interface for a GPU kernel.
  */
class Kernel
{
public:
	virtual ~Kernel()
	{
	}

	/**
	 * @brief Returns the kernel's unique name.
	 */
	virtual std::string name() const = 0;

	/**
	 * @brief Launches the kernel on the given inputs.
	 * @param[in]  input_a  First input vector.
	 * @param[in]  input_b  Second input vector.
	 * @param[out] output   Output vector (must be preallocated to size N).
	 */
	virtual void launch(const std::vector<float>& input_a, const std::vector<float>& input_b,
		std::vector<float>& output) = 0;
};