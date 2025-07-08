/**
 * @file    {{name}}.h
 * @brief   Host launchers for different {{name}} kernel variants (global, shared memory, float4).
 * @author  Kamil J.
 * @date    {{date}}
 *
 * Contains host functions to launch various {{name}} kernels for benchmarking.
 */

#pragma once

/**
 * @brief Runs the global memory {{name}} kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_
{
    {
        name
    }
}
_global(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the shared memory {{name}} kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_
{
    {
        name
    }
}
_shared(const float* a, const float* b, float* c, int N);

/**
 * @brief Runs the float4 vectorized {{name}} kernel.
 *
 * @param[in]  a  Pointer to the first input array.
 * @param[in]  b  Pointer to the second input array.
 * @param[out] c  Pointer to the output array.
 * @param[in]  N  Number of elements.
 * @return        Kernel execution time in milliseconds.
 */
extern "C" float run_
{
    {
        name
    }
}
_float4(const float* a, const float* b, float* c, int N);