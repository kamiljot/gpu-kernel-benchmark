/**
 * @file    add.h
 * @brief   Host launchers for the elementwise add operation (global, shared, float4).
 * @author  Kamil J.
 * @date    2025-07-07
 *
 * This header declares host-side launcher functions used to execute and measure
 * the elementwise addition operation on the CPU and several GPU kernel variants.
 */

#pragma once

// Lightweight forward declaration to avoid including heavy headers in kernel headers
enum class BenchmarkMode;

/**
 * @brief Launches the global-memory add kernel and returns measured execution time.
 *
 * Behavior:
 *  - Allocates device memory for inputs and output using cudaMalloc.
 *  - Copies input arrays from host to device.
 *  - Launches the kernel that reads inputs from global memory and writes outputs to global memory.
 *  - Copies the output back to host and frees device memory.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Measured kernel execution time in milliseconds, or negative value on error.
 */
extern "C" float run_add_global(const float* a, const float* b, float* c, int N);

/**
 * @brief Launches the shared-memory add kernel and returns measured execution time.
 *
 * Behavior:
 *  - Allocates device memory and copies inputs.
 *  - Launches a kernel that stages per-block tiles of input into shared memory for improved locality.
 *  - Copies the output back to host and frees device memory.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Measured kernel execution time in milliseconds, or negative value on error.
 */
extern "C" float run_add_shared(const float* a, const float* b, float* c, int N);

/**
 * @brief Launches the float4-vectorized add kernel and returns measured execution time.
 *
 * Behavior:
 *  - Packs scalar input arrays into float4 vectors (host-side) and pads to a multiple of 4.
 *  - Allocates device memory for packed float4 arrays and copies them to the device.
 *  - Launches a kernel where each thread processes one float4 element (4 scalars).
 *  - Copies results back to host and unpacks into the scalar output array.
 *
 * Requirements:
 *  - N may be not divisible by 4; the launcher pads the input so this function works for any N.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of scalar elements.
 * @return        Measured kernel execution time in milliseconds, or negative value on error.
 */
extern "C" float run_add_float4(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the global-memory add kernel using preallocated device buffers (persistent buffer) and returns measured execution time.
 *
 * This function is intended for internal dispatchers that manage a PersistentBuffer and want
 * to avoid repeated cudaMalloc/cudaFree across invocations. Depending on @p mode, the
 * function either measures only kernel execution (KernelOnly) or the full H2D+kernel+D2H sequence (EndToEnd).
 *
 * @param[in]  a      Host pointer to the first input array.
 * @param[in]  b      Host pointer to the second input array.
 * @param[out] c      Host pointer to the output array.
 * @param[in]  N      Number of elements.
 * @param[in]  mode   BenchmarkMode controlling measurement scope.
 * @return            Measured time in milliseconds, or negative value on error.
 */
float run_add_global_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode);

/**
 * @brief Runs the shared-memory add kernel using preallocated device buffers (persistent buffer) and returns measured execution time.
 *
 * See `run_add_global_with_buffer` for behavior regarding BenchmarkMode and buffer reuse.
 * This variant launches the shared-memory kernel implementation.
 *
 * @param[in]  a      Host pointer to the first input array.
 * @param[in]  b      Host pointer to the second input array.
 * @param[out] c      Host pointer to the output array.
 * @param[in]  N      Number of elements.
 * @param[in]  mode   BenchmarkMode controlling measurement scope.
 * @return            Measured time in milliseconds, or negative value on error.
 */
float run_add_shared_with_buffer(const float* a, const float* b, float* c, int N, BenchmarkMode mode);
