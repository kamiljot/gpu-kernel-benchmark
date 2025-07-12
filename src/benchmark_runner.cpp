/**
 * @file    benchmark_runner.cpp
 * @brief   Main benchmark runner for modular GPU kernel benchmarking framework.
 * @author  Kamil J.
 * @date    2025-07-10
 */

#include "backend/backend_registry.hpp"
#include "benchmark/cli_args.hpp"
#include "benchmark/dispatch_benchmarks.hpp"
#include "benchmark/generators.hpp"
#include "benchmark/selectors.hpp"
#include "kernels/kernel_registry.hpp"
#include "utils/csv_logger.hpp"

extern void force_backend_registration();
extern void force_kernel_registration();

int main(int argc, char* argv[])
{
    force_backend_registration();
    force_kernel_registration();
    Args args = parse_args(argc, argv);

    auto backends = select_backends(args.backend);
    auto kernels = select_kernels(args.op, args.variant);

    CsvLogger logger(args.csv_file, {"backend", "kernel", "size", "pass", "cpu_time_ms", "transfer_in_ms", "kernel_ms",
                                     "transfer_out_ms"});
    dispatch_benchmarks(args, backends, kernels, logger);
    return 0;
}
