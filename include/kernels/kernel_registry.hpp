/**
 * @file    kernel_registry.hpp
 * @brief   Factory and registry for all available kernels.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Provides dynamic registration and creation of kernels by name.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <vector>
#include "kernel_interface.hpp"

 /**
  * @class   KernelRegistry
  * @brief   Factory and registry for all available kernels.
  *
  * Allows dynamic registration and creation of kernels by name.
  */
class KernelRegistry {
public:
	/// Type for kernel creator function.
	using Creator = std::function<std::unique_ptr<KernelInterface>()>;

	/**
	 * @brief Singleton accessor.
	 * @return Reference to the global KernelRegistry.
	 */
	static KernelRegistry& instance() {
		static KernelRegistry registry;
		return registry;
	}

	/**
	 * @brief Register a kernel creator for a specific operation name.
	 *
	 * @param name    Name of the operation (e.g. "add").
	 * @param creator Functor that returns a new kernel instance.
	 */
	void register_kernel(const std::string& name, Creator creator) {
		creators_[name] = std::move(creator);
	}

	/**
	 * @brief Create a new instance of a registered kernel.
	 *
	 * @param name Name of the kernel to create.
	 * @return     Unique pointer to the created kernel, or nullptr if not found.
	 */
	std::unique_ptr<KernelInterface> create(const std::string& name) const {
		auto it = creators_.find(name);
		if (it != creators_.end())
			return (it->second)();
		return nullptr;
	}

	/**
	 * @brief Get a list of all registered kernel operation names.
	 * @return Vector of registered kernel names.
	 */
	std::vector<std::string> available_kernels() const {
		std::vector<std::string> result;
		for (const auto& kv : creators_) result.push_back(kv.first);
		return result;
	}

private:
	std::unordered_map<std::string, Creator> creators_; ///< Map: kernel name -> factory function
};