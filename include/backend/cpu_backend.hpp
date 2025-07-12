/**
 * @file    cpu_backend.hpp
 * @brief   CPU backend for kernel benchmark framework.
 * @author  Kamil J.
 * @date    2025-07-11
 */

#pragma once

#include <string>

#include "backend/backend_interface.hpp"

/**
 * @brief CPU backend implementation.
 */
class CpuBackend : public BackendInterface
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
};