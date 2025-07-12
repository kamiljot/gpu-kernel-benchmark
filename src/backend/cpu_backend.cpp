#include "backend/cpu_backend.hpp"

#include "kernels/kernel_registry.hpp"
#include "utils/time_utils.hpp"

std::string CpuBackend::name() const
{
    return "cpu";
}

float* CpuBackend::allocate(size_t num_elements)
{
    // CPU: just allocate host memory
    return new float[num_elements];
}

void CpuBackend::free(float* ptr)
{
    delete[] ptr;
}

void CpuBackend::copy_to_device(float* dst, const float* src, size_t num_elements)
{
    // CPU: direct copy
    std::copy(src, src + num_elements, dst);
}

void CpuBackend::copy_to_host(float* dst, const float* src, size_t num_elements)
{
    // CPU: direct copy
    std::copy(src, src + num_elements, dst);
}

void CpuBackend::launch_kernel(const std::string& kernel_name, float* /*d_in1*/, float* /*d_in2*/, float* /*d_out*/,
                               const float* h_in1, const float* h_in2, float* h_out, size_t size, GpuTiming* timing)
{
    auto kernel = KernelRegistry::instance().create(kernel_name);
    CpuTimer timer;
    timer.start();
    kernel->run(const_cast<float*>(h_in1), const_cast<float*>(h_in2), h_out, size);
    timer.stop();
    if (timing)
    {
        timing->kernel_ms = timer.elapsed_ms();
        timing->transfer_in_ms = 0.0f;
        timing->transfer_out_ms = 0.0f;
    }
}