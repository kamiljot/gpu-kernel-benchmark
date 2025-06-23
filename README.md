# GPU Kernel Benchmark

This project benchmarks and compares multiple CUDA kernel implementations (global memory, shared memory, and float4 vectorization) for a variety of mathematical operations on large arrays.

The goal is to evaluate performance improvements gained from memory access optimization strategies on the GPU.

---

## Features

- Modular structure for benchmarking different operations (e.g. sqrt + log, add, mul)
- Three GPU variants per operation: global memory, shared memory, float4
- Automatic CSV logging and plotting scripts
- Customizable number of benchmark passes and input sizes
- Python tools for visualizing performance and speedup

---

## Directory Structure

```
gpu-kernel-benchmark/
??? include/                 # C++ headers
??? src/                     # Source files (.cpp, .cu)
?   ??? gpu_kernel.cu        # Default kernel (sqrt + log)
?   ??? gpu_memory_utils.cu  # Memory allocation helpers
?   ??? gpu_benchmark_utils.cu # Timing logic for each kernel variant
?   ??? benchmark_utils.cpp  # Input generation & CSV output
?   ??? main.cpp             # Main benchmarking loop
??? benchmarks/              # Output CSV + plots
??? plot_float4_compare.py   # Full distribution boxplot
??? plot_float4_compare_avg.py # Mean of best 10% line plot
??? README.md
??? BUILD.md
??? CMakeLists.txt
```

---

## Build Instructions

See [BUILD.md](./BUILD.md) for detailed build steps.

Basic steps:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

---

## Run Benchmark

```bash
./gpu_kernel_benchmark 100
```
- The argument `100` sets the number of passes per input size (default is 100).
- Results are saved to: `benchmarks/results_all.csv`

---

## Plotting Results

Install Python dependencies:

```bash
pip install pandas matplotlib seaborn
```

## Run:
```bash
python plot_float4_compare.py       # boxplot of all measurements
python plot_float4_compare_avg.py   # average of best 10% only
```

## Output files:
- `benchmarks/exec_time_float4.png`
- `benchmarks/speedup_float4.png`
- `benchmarks/exec_time_best10.png`
- `benchmarks/speedup_best10.png`

---

## TODO
- [x] Separate kernel variants into modular files
- [x] Support float4 vectorized memory access
- [x] Add per-kernel benchmark entry points
- [x] Record full and trimmed timings
- [x] Include plotting and analysis scripts
- [ ] Benchmark additional ops: add, mul, exp+sin, etc.
- [ ] Add CLI selector for kernel and variant
- [ ] Generate HTML performance reports