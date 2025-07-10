/**
 * @file    backend_interface.hpp
 * @brief   Abstract interface for compute backends (CPU, CUDA, etc.)
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Defines the API for backend abstraction (memory, kernel launches).
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

 /**
  * @class   BackendInterface
  * @brief   Interface for compute device backend (CPU, CUDA, etc.)
  *
  * Provides abstraction for memory management and kernel launches.
  */
class BackendInterface {
public:
	virtual ~BackendInterface() = default;

	/**
	 * @brief Get the name of the backend.
	 * @return Backend name (e.g. "cpu", "cuda")
	 */
	virtual std::string name() const = 0;

	/**
	 * @brief Allocate device memory for floats.
	 * @param num_elements Number of elements to allocate.
	 * @return Pointer to device memory, or nullptr on error.
	 */
	virtual float* allocate(size_t num_elements) = 0;

	/**
	 * @brief Free device memory allocated with allocate().
	 * @param ptr Pointer to device memory.
	 */
	virtual void free(float* ptr) = 0;

	/**
	 * @brief Copy memory from host to device.
	 * @param dst Device pointer.
	 * @param src Host pointer.
	 * @param num_elements Number of elements to copy.
	 */
	virtual void copy_to_device(float* dst, const float* src, size_t num_elements) = 0;

	/**
	 * @brief Copy memory from device to host.
	 * @param dst Host pointer.
	 * @param src Device pointer.
	 * @param num_elements Number of elements to copy.
	 */
	virtual void copy_to_host(float* dst, const float* src, size_t num_elements) = 0;

	/**
	 * @brief Launch the given kernel on this backend (can be a dispatcher).
	 * @param kernel_name Name of the registered kernel (e.g. "add").
	 * @param input1 First input pointer (device).
	 * @param input2 Second input pointer (device or nullptr).
	 * @param output Output pointer (device).
	 * @param size Number of elements.
	 */
	virtual void launch_kernel(const std::string& kernel_name,
		float* input1, float* input2, float* output, size_t size) = 0;
};