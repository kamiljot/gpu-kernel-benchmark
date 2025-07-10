/**
 * @file    register_backends.cpp
 * @brief   Registers all compute backends in the global BackendRegistry.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Performs static registration of all compute backends.
 */

#include "../include/backend/backend_registry.hpp"
#include "cpu_backend.hpp"
 // #include "backend/cuda_backend.hpp" // For future CUDA backend

namespace {
	struct StaticBackendRegistrations {
		StaticBackendRegistrations() {
			BackendRegistry::instance().register_backend("cpu", [] {
				return std::make_unique<CpuBackend>();
				});
			// BackendRegistry::instance().register_backend("cuda", [] {
			//     return std::make_unique<CudaBackend>();
			// });
		}
	};
	static StaticBackendRegistrations registrations;
}

// **Force registration function**
void force_backend_registration() {}