/**
 * @file    register_kernels.cpp
 * @brief   Registers all available kernels in the KernelRegistry.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Performs static registration of all modular kernel variants.
 */

#include "kernels/kernel_registry.hpp"
#include "kernels/add/add_kernel.hpp"

namespace {
	/**
	 * @brief Registers available kernel implementations at startup.
	 */
	struct StaticKernelRegistrations {
		StaticKernelRegistrations() {
			KernelRegistry::instance().register_kernel("add_global", [] {
				return std::make_unique<AddGlobalKernel>();
				});
			// TODO: register other variants/kernels (e.g., shared, float4, etc.)
		}
	};
	static StaticKernelRegistrations registrations;
}

/**
 * @brief Dummy function to force translation unit inclusion.
 */
void force_kernel_registration() {}