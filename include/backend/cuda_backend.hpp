/**
 * @file    cuda_backend.hpp
 * @brief   CUDA backend for kernel benchmark framework.
 * @author  Kamil J.
 * @date    2025-07-11
 */

#pragma once

#include <string>

#include "backend/backend_interface.hpp"

/**
 * @brief CUDA backend implementation.
 */
class CudaBackend : public BackendInterface
{
   public:
    std::string name() const override;
    float* allocate(size_t num_elements) override;
    void free(float* ptr) override;
    void copy_to_device(float* dst, const float* src, size_t num_elements) override;
    void copy_to_host(float* dst, const float* src, size_t num_elements) override;

    /**
     * @copydoc BackendInterface::launch_kernel()
     */
    void launch_kernel(const std::string& kernel_name, float* d_in1, float* d_in2, float* d_out, const float* h_in1,
                       const float* h_in2, float* h_out, size_t size, GpuTiming* timing) override;

   private:
    void measure_transfer_in(float* d_in1, float* d_in2, const float* h_in1, const float* h_in2, size_t size,
                             float& elapsed_ms);
    void measure_kernel(const std::string& kernel_name, float* d_in1, float* d_in2, float* d_out, size_t size,
                        float& elapsed_ms);
    void measure_transfer_out(float* h_out, float* d_out, size_t size, float& elapsed_ms);
};