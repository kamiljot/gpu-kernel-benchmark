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
 * @brief Allocates GPU memory for input arrays a, b, and output array c,
 *        and copies input data from host to device.
 *
 * @param[in]  a    Host pointer to the first input array.
 * @param[in]  b    Host pointer to the second input array.
 * @param[out] d_a  Device pointer to the allocated first input array.
 * @param[out] d_b  Device pointer to the allocated second input array.
 * @param[out] d_c  Device pointer to the allocated output array.
 * @param[in]  N    Number of elements.
 */
void allocate_and_copy(const float* a, const float* b, float** d_a, float** d_b, float** d_c, int N);

/**
 * @brief Allocates GPU memory for float4 input arrays a, b, and output array c,
 *        and copies input data (already packed into float4) from host to device.
 *
 * @param[in]  a      Host pointer to the first input array (packed as float4).
 * @param[in]  b      Host pointer to the second input array (packed as float4).
 * @param[out] d_a4   Device pointer to the allocated first input array (float4).
 * @param[out] d_b4   Device pointer to the allocated second input array (float4).
 * @param[out] d_c4   Device pointer to the allocated output array (float4).
 * @param[in]  N_vec4 Number of float4 elements.
 */
void allocate_and_copy_vec4(const float* a, const float* b, float4** d_a4, float4** d_b4, float4** d_c4, int N_vec4);

/**
 * @brief Frees GPU memory for device arrays.
 *
 * @param[in] d_a  Device pointer to the first input array.
 * @param[in] d_b  Device pointer to the second input array.
 * @param[in] d_c  Device pointer to the output array.
 */
void free_device(float* d_a, float* d_b, float* d_c);