/**
 * @file    register_kernels.cpp
 * @brief   Registers all kernels in the global KernelRegistry.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Performs static registration of all kernels.
 */

#include "../include/kernels/kernel_registry.hpp"
#include "add/add_kernel.hpp"
 // #include "kernels/sin_cos_pow_relu/sin_cos_pow_relu_kernel.hpp" // Example

namespace {
	/**
	 * @brief Static block to register all kernels.
	 */
	struct StaticKernelRegistrations {
		StaticKernelRegistrations() {
			KernelRegistry::instance().register_kernel("add", [] {
				return std::make_unique<AddKernel>();
				});
			// KernelRegistry::instance().register_kernel("sin_cos_pow_relu", [] {
			//     return std::make_unique<SinCosPowReluKernel>();
			// });
			// Register more kernels here as needed.
		}
	};
	static StaticKernelRegistrations registrations;
}