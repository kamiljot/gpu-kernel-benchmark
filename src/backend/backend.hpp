/**
 * @file    backend.hpp
 * @brief   Abstract backend interface for modular GPU benchmarking.
 * @author  Kamil J.
 * @date    2025-07-09
 */

#pragma once
#include <string>
#include <vector>

class Kernel;  ///< Forward declaration

/**
 * @class   Backend
 * @brief   Abstract interface for all GPU backends (CUDA, ...).
 */
class Backend
{
public:
	virtual ~Backend()
	{
	}

	/**
	 * @brief Returns the backend name (e.g., "cuda").
	 */
	virtual std::string name() const = 0;

	/**
	 * @brief Initializes the backend.
	 */
	virtual void initialize() = 0;

	/**
	 * @brief Loads or prepares the kernel.
	 * @param[in]  kernel_name  Name of the kernel to load.
	 */
	virtual void load_kernel(const std::string& kernel_name) = 0;

	/**
	 * @brief Launches the kernel using backend-specific logic.
	 * @param[in]   kernel      Pointer to kernel object.
	 * @param[in]   input_a     First input vector.
	 * @param[in]   input_b     Second input vector.
	 * @param[out]  output      Output vector.
	 */
	virtual void launch(Kernel* kernel, const std::vector<float>& input_a, const std::vector<float>& input_b,
		std::vector<float>& output) = 0;

	/**
	 * @brief Returns the last measured execution time in milliseconds.
	 */
	virtual double get_last_execution_time() const = 0;
};
