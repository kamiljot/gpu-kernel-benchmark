/**
 * @file    backend_interface.hpp
 * @brief   Abstract interface for all compute backends (CPU/GPU).
 * @author  Kamil J.
 * @date    2025-07-11
 */

#pragma once

#include <string>

#include "backend/gpu_timing.hpp"

/**
 * @brief Interface for all compute backends (CPU, CUDA, ...).
 */
class BackendInterface
{
   public:
    virtual ~BackendInterface() = default;

    /**
     * @brief Get backend name (e.g. "cpu" or "cuda").
     */
    virtual std::string name() const = 0;

    /**
     * @brief Allocate device memory.
     */
    virtual float* allocate(size_t num_elements) = 0;

    /**
     * @brief Free device memory.
     */
    virtual void free(float* ptr) = 0;

    /**
     * @brief Copy data from host to device.
     */
    virtual void copy_to_device(float* dst, const float* src, size_t num_elements) = 0;

    /**
     * @brief Copy data from device to host.
     */
    virtual void copy_to_host(float* dst, const float* src, size_t num_elements) = 0;

    /**
     * @brief Launch a registered kernel on the backend.
     *
     * @param kernel_name Name of the kernel.
     * @param d_in1 Device pointer to first input array.
     * @param d_in2 Device pointer to second input array.
     * @param d_out Device pointer to output array.
     * @param h_in1 Host pointer to first input (can be nullptr if not needed).
     * @param h_in2 Host pointer to second input (can be nullptr if not needed).
     * @param h_out Host pointer to output (can be nullptr if not needed).
     * @param size Number of elements.
     * @param timing Pointer to GpuTiming struct for performance measurement (can be nullptr).
     */
    virtual void launch_kernel(const std::string& kernel_name, float* d_in1, float* d_in2, float* d_out,
                               const float* h_in1, const float* h_in2, float* h_out, size_t size,
                               GpuTiming* timing) = 0;
};
