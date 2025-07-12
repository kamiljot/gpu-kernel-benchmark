/**
 * @file    gpu_timing.hpp
 * @brief   Structure for storing GPU timing information.
 * @author  Kamil J.
 * @date    2025-07-11
 */
#pragma once

/**
 * @struct  GpuTiming
 * @brief   Records timing for memory transfers and kernel execution on GPU.
 */
struct GpuTiming
{
    float transfer_in_ms = 0;
    float kernel_ms = 0;
    float transfer_out_ms = 0;
};