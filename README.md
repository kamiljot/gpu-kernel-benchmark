# GPU Kernel Benchmark

This project benchmarks multiple GPU kernel implementations for common math operations on large arrays, comparing CPU performance against different GPU memory access strategies: global memory, shared memory, and float4 vectorized memory.

## Features

- Modular design with separate kernel, launcher, and utility files
- Support for different operations: `sqrt+log`, `add` (more to come)
- Three GPU variants per operation:
  - Global memory access
  - Shared memory
  - float4 vectorization
- Automatic input data generation if missing
- CSV logging of benchmark results
- Python scripts for plotting results
- Optional batch mode benchmarking with configurable passes

## Project Structure

```
gpu-kernel-benchmark/
├── include/                       # Header files
│   ├── add.h
│   ├── benchmark_utils.h
│   ├── cpu_baseline.h
│   ├── gpu_benchmark_utils.h
│   ├── gpu_memory_utils.h
│   ├── input_generator.h
│   ├── kernel_dispatch.h
│   ├── sqrt_log.h
│   └── timer.hpp
│
├── src/                          # Source files
│   ├── main.cpp                  # Single-run benchmark mode
│   ├── benchmark_batch.cpp       # Batch mode: run many samples per size
│   ├── benchmark_utils.cpp       # CSV and input helpers
│   ├── cpu_baseline.cpp          # CPU fallback version
│   ├── gpu_memory_utils.cu       # GPU malloc/copy utilities
│   ├── input_generator.cpp       # Input data generation
│   ├── kernel_dispatch.cpp       # Dispatch logic for kernel variants
│   ├── cuda_utils.cuh            # Shared CUDA macros/utilities
│   └── kernels/
│       ├── add/
│       │   ├── add_kernels.cu
│       │   ├── add_kernels.cuh
│       │   └── add_launcher.cu
│       └── sqrt_log/
│           ├── sqrt_log_kernels.cu
│           ├── sqrt_log_kernels.cuh
│           └── sqrt_log_launcher.cu
│
├── benchmarks/                   # Output CSVs and plots
│   ├── result.csv
│   ├── exec_time_float4.png
│   ├── exec_time_best10.png
│   ├── speedup_float4.png
│   └── speedup_best10.png
│
├── plot_float4_compare.py        # Full boxplot comparison
├── plot_float4_compare_avg.py    # Best 10% mean line plot
├── README.md
├── BUILD.md                      # Build instructions
└── CMakeLists.txt
```

## Build Instructions

See [BUILD.md](./BUILD.md) for full setup. Minimal steps:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Run Benchmark (Single Pass)

```bash
./gpu_kernel_benchmark <operation> <input_file>
```

- If `<input_file>` does not exist, it will be auto-generated.
- Example:

```bash
./gpu_kernel_benchmark add input_file
```

## Run Benchmark (Batch Mode)

```bash
./gpu_kernel_batch <operation> <passes>
```

- Example: run `100` passes for `sqrt_log`:

```bash
./gpu_kernel_batch sqrt_log 100
```

Results saved to `benchmarks/result.csv`

## Plot Results

Install Python dependencies:

```bash
pip install pandas matplotlib seaborn
```

Generate charts:

```bash
python plot_float4_compare.py       # Boxplot
python plot_float4_compare_avg.py   # Best 10% average plot
```

Output files:
- `benchmarks/exec_time_float4.png`
- `benchmarks/exec_time_best10.png`
- `benchmarks/speedup_float4.png`
- `benchmarks/speedup_best10.png`

## TODO

- [x] Modular file separation by operation
- [x] Benchmark shared memory and float4
- [x] Add batch mode benchmark
- [x] Add input auto-generation
- [x] Add Python scripts for analysis
- [ ] Add more math ops (mul, sin+exp, etc.)
- [ ] CLI flag for selecting individual kernel variant
- [ ] JSON/HTML report export