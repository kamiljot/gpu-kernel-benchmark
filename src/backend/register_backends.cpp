/**
 * @file    register_backends.cpp
 * @brief   Registers all compute backends in the global BackendRegistry.
 * @author  Kamil J.
 * @date    2025-07-10
 *
 * Performs static registration of all compute backends.
 */

#include "backend/backend_registry.hpp"
#include "backend/cpu_backend.hpp"
//#ifdef USE_CUDA
#include "backend/cuda_backend.hpp"
//#endif

//#ifdef USE_CUDA
//#pragma message("CUDA backend registration compiled in!")
//#endif

namespace
{
struct StaticBackendRegistrations
{
    StaticBackendRegistrations()
    {
        BackendRegistry::instance().register_backend("cpu", [] { return std::make_unique<CpuBackend>(); });
//#ifdef USE_CUDA
        BackendRegistry::instance().register_backend("cuda", [] { return std::make_unique<CudaBackend>(); });
//#endif
    }
};
static StaticBackendRegistrations registrations;
}  // namespace

// **Force registration function**
void force_backend_registration()
{
}