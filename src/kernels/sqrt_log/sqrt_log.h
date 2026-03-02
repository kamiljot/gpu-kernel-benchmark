/**
 * @file    sqrt_log.h
 * @brief   Host launchers for sqrt_log kernel variants (global, shared memory, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * Contains host functions to launch sqrt_log kernels for benchmarking and validation.
 */

#pragma once

// Lightweight forward declaration to avoid transitive includes in kernel headers
enum class BenchmarkMode;

/**
 * @brief Launches the global-memory sqrt_log kernel and returns measured execution time.
 *
 * The kernel computes c[i] = sqrtf(a[i]) + logf(b[i] + 1e-6f) for each element.
 * This launcher performs device allocation, copies inputs to device, launches the kernel,
 * copies results back to host, and frees device memory.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds, or negative value on error.
 */
extern "C" float run_sqrt_log_global(const float* a, const float* b, float* c, int N);

/**
 * @brief Launches the shared-memory sqrt_log kernel and returns measured execution time.
 *
 * This variant stages per-block tiles of inputs into shared memory to improve memory locality
 * before computing the same sqrt/log expression as the global variant.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds, or negative value on error.
 */
extern "C" float run_sqrt_log_shared(const float* a, const float* b, float* c, int N);

/**
 * @brief Launches the float4-vectorized sqrt_log kernel and returns measured execution time.
 *
 * Host-side launcher packs inputs into float4 vectors, copies them to device, launches a
 * vectorized kernel that processes four scalar elements per thread, and unpacks results.
 *
 * Requirements:
 *  - Input arrays will be padded to a multiple of 4; behaviour is defined for any N.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of scalar elements.
 * @return        Kernel execution time in milliseconds, or negative value on error.
 */
extern "C" float run_sqrt_log_float4(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the global-memory sqrt_log kernel using a persistent device buffer and returns measured execution time.
 *
 * This helper reuses previously allocated device buffers to avoid repeated cudaMalloc/cudaFree.
 * When called with BenchmarkMode::KernelOnly it measures only kernel execution; with
 * BenchmarkMode::EndToEnd it includes host->device and device->host copies in the timing.
 *
 * @param[in]  a      Host pointer to first input array.
 * @param[in]  b      Host pointer to second input array.
 * @param[out] c      Host pointer to output array.
 * @param[in]  N      Number of elements.
 * @param[in]  mode   Measurement mode (KernelOnly or EndToEnd).
 * @return            Measured time in milliseconds, or negative value on error.
 */
float run_sqrt_log_global_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode);

/**
 * @brief Runs the shared-memory sqrt_log kernel using a persistent device buffer and returns measured execution time.
 *
 * Identical semantics to `run_sqrt_log_global_with_buffer` but launches the shared-memory variant.
 *
 * @param[in]  a      Host pointer to first input array.
 * @param[in]  b      Host pointer to second input array.
 * @param[out] c      Host pointer to output array.
 * @param[in]  N      Number of elements.
 * @param[in]  mode   Measurement mode (KernelOnly or EndToEnd).
 * @return            Measured time in milliseconds, or negative value on error.
 */
float run_sqrt_log_shared_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode);
