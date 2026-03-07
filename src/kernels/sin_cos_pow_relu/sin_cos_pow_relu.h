/**
 * @file    sin_cos_pow_relu.h
 * @brief   Host launchers for sin_cos_pow_relu kernel variants (global, shared memory, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains host functions to launch various sin_cos_pow_relu kernels for benchmarking.
 */

#pragma once

// Forward declaration to avoid pulling in larger headers from this lightweight kernel header
enum class BenchmarkMode;

/**
 * @brief Runs the global memory sin_cos_pow_relu kernel and returns measured execution time.
 *
 * The function performs:
 *  - Device allocation and host-to-device copies for inputs.
 *  - A single timed kernel launch that uses global memory for inputs/outputs.
 *  - Device-to-host copy of the results and cleanup of device allocations.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_sin_cos_pow_relu_global(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the shared memory sin_cos_pow_relu kernel and returns measured execution time.
 *
 * The function performs:
 *  - Device allocation and host-to-device copies for inputs.
 *  - Launches the kernel that stages inputs into shared memory per block for improved locality.
 *  - Copies results back to host and frees device allocations.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_sin_cos_pow_relu_shared(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the float4 vectorized sin_cos_pow_relu kernel and returns measured execution time.
 *
 * The function performs:
 *  - Packs input arrays into float4 vectors (host-side) and pads if necessary.
 *  - Allocates device memory for packed vectors and copies data to device.
 *  - Launches the vectorized kernel that processes four floats per thread.
 *  - Copies results back to host, unpacks into scalar output, and frees device memory.
 *
 * Requirements:
 *  - The input size N must be divisible by 4; otherwise std::invalid_argument may be thrown by the launcher.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_sin_cos_pow_relu_float4(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the global memory sin_cos_pow_relu kernel using a persistent device buffer and returns measured execution time.
 *
 * The function performs:
 *  - Allocates and reuses device memory for inputs/outputs across launches (persistent buffer).
 *  - Copies input arrays to device memory as needed (depending on BenchmarkMode).
 *  - Launches the global memory kernel for sin_cos_pow_relu.
 *  - Measures kernel or end-to-end time using CUDA events.
 *  - Copies results back to host.
 *
 * Requirements:
 *  - The persistent buffer is reused for multiple launches to avoid repeated cudaMalloc/cudaFree.
 *
 * @param[in]  a      Pointer to the first input array.
 * @param[in]  b      Pointer to the second input array.
 * @param[out] c      Pointer to the output array.
 * @param[in]  N      Number of elements.
 * @param[in]  mode   Benchmark measurement mode (KernelOnly or EndToEnd).
 * @return            Kernel execution time in milliseconds.
 */
float run_sin_cos_pow_relu_global_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode);

/**
 * @brief Runs the shared memory sin_cos_pow_relu kernel using a persistent device buffer and returns measured execution time.
 *
 * The function performs:
 *  - Allocates and reuses device memory for inputs/outputs across launches (persistent buffer).
 *  - Copies input arrays to device memory as needed (depending on BenchmarkMode).
 *  - Launches the shared memory kernel for sin_cos_pow_relu.
 *  - Measures kernel or end-to-end time using CUDA events.
 *  - Copies results back to host.
 *
 * Requirements:
 *  - The persistent buffer is reused for multiple launches to avoid repeated cudaMalloc/cudaFree.
 *
 * @param[in]  a      Pointer to the first input array.
 * @param[in]  b      Pointer to the second input array.
 * @param[out] c      Pointer to the output array.
 * @param[in]  N      Number of elements.
 * @param[in]  mode   Benchmark measurement mode (KernelOnly or EndToEnd).
 * @return            Kernel execution time in milliseconds.
 */
float run_sin_cos_pow_relu_shared_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode);
