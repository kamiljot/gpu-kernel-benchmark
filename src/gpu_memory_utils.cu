/**
 * @file    gpu_memory_utils.cu
 * @brief   Implements device memory allocation, deallocation, and copy routines for GPU arrays.
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Provides implementations for allocating, copying, and freeing device memory
 * for standard and float4 arrays in CUDA-based benchmarks.
 */

#include "gpu_memory_utils.h"

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
void allocate_and_copy(const float* a, const float* b, float** d_a, float** d_b, float** d_c, int N)
{
    size_t size = N * sizeof(float);
    cudaMalloc(d_a, size);
    cudaMalloc(d_b, size);
    cudaMalloc(d_c, size);
    cudaMemcpy(*d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b, b, size, cudaMemcpyHostToDevice);
}

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
void allocate_and_copy_vec4(const float* a, const float* b, float4** d_a4, float4** d_b4, float4** d_c4, int N_vec4)
{
    size_t size = N_vec4 * sizeof(float4);
    cudaMalloc(d_a4, size);
    cudaMalloc(d_b4, size);
    cudaMalloc(d_c4, size);
    cudaMemcpy(*d_a4, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_b4, b, size, cudaMemcpyHostToDevice);
}

/**
 * @brief Frees GPU memory for device arrays.
 *
 * @param[in] d_a  Device pointer to the first input array.
 * @param[in] d_b  Device pointer to the second input array.
 * @param[in] d_c  Device pointer to the output array.
 */
void free_device(float* d_a, float* d_b, float* d_c)
{
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}