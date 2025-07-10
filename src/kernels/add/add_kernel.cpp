/**
 * @file    add_kernel.cpp
 * @brief   Implementation of AddGlobalKernel (CPU/GPU dispatch).
 * @author  Kamil J.
 * @date    2025-07-10
 */

#include "kernels/add/add_kernel.hpp"

#ifdef USE_CUDA
#include "backend/backend_registry.hpp"
#include "kernels/add/add_kernel_cuda.hpp" // extern "C" void launch_cuda_add_global(...)
#endif

void AddGlobalKernel::run(float* in1, float* in2, float* out, size_t size) {
#ifdef USE_CUDA
	// Example: use CUDA if current backend is CUDA
	auto backend = BackendRegistry::instance().get_current();
	if (backend && backend->name() == "cuda") {
		launch_cuda_add_global(in1, in2, out, size);
		return;
	}
#endif
	// Default: CPU fallback
	for (size_t i = 0; i < size; ++i)
		out[i] = in1[i] + in2[i];
}