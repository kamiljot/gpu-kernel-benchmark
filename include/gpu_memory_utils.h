/**
 * @file    gpu_memory_utils.h
 * @brief   Utilities for CUDA memory allocation, copy, and free operations.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains helper functions for managing device memory in CUDA benchmarks.
 */

#pragma once
#include <cuda_runtime.h>

/**
 * @brief Allocate device buffers for inputs and output and copy input data from host.
 *
 * This helper wraps repeated cudaMalloc and cudaMemcpy operations used by launchers.
 * On success, device pointers are returned via the output parameters and must be
 * freed by the caller (use free_device).
 *
 * @param[in]  a    Host pointer to the first input array (size N).
 * @param[in]  b    Host pointer to the second input array (size N).
 * @param[out] d_a  Pointer to receive device pointer for first input (allocated inside).
 * @param[out] d_b  Pointer to receive device pointer for second input (allocated inside).
 * @param[out] d_c  Pointer to receive device pointer for output (allocated inside).
 * @param[in]  N    Number of elements to allocate for each array.
 */
void allocate_and_copy(const float* a, const float* b, float** d_a, float** d_b, float** d_c, int N);

/**
 * @brief Allocate device buffers for float4-packed inputs and copy packed data to device.
 *
 * This helper assumes host arrays are already packed as consecutive float4 elements.
 * It performs cudaMalloc for float4 buffers and copies the data to device memory.
 *
 * @param[in]  a      Host pointer to the first input array (packed as float4, length N_vec4).
 * @param[in]  b      Host pointer to the second input array (packed as float4, length N_vec4).
 * @param[out] d_a4   Pointer to receive device pointer for first input (allocated inside).
 * @param[out] d_b4   Pointer to receive device pointer for second input (allocated inside).
 * @param[out] d_c4   Pointer to receive device pointer for output (allocated inside).
 * @param[in]  N_vec4 Number of float4 elements to allocate and copy.
 */
void allocate_and_copy_vec4(const float* a, const float* b, float4** d_a4, float4** d_b4, float4** d_c4, int N_vec4);

/**
 * @brief Frees device memory allocated for the three arrays.
 *
 * This is a convenience wrapper around cudaFree for the three allocated buffers.
 * Call this helper to release device resources allocated by allocate_and_copy.
 *
 * @param[in] d_a  Device pointer to the first input array (may be nullptr).
 * @param[in] d_b  Device pointer to the second input array (may be nullptr).
 * @param[in] d_c  Device pointer to the output array (may be nullptr).
 */
void free_device(float* d_a, float* d_b, float* d_c);
