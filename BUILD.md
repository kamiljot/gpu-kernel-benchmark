# Build Instructions for GPU Kernel Benchmark

This document explains how to build and run the benchmark project on different platforms.

---

## Requirements

- CUDA Toolkit 11.x or newer
- CMake ≥ 3.10
- C++17-compatible compiler:
  - Linux/macOS: GCC or Clang
  - Windows: MSVC (Visual Studio 2022 or newer)
- Python 3.x with the following packages:
  - `pandas`
  - `matplotlib`
  - `seaborn`

---

## Build Steps

### Linux / macOS / WSL

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Windows (Visual Studio)

1. Open "x64 Native Tools Command Prompt for VS 2022"
2. Run:

```cmd
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

---

## Run the Benchmark

From the `build/` directory:

```bash
./gpu_kernel_benchmark 100
```

- Replace `100` with the number of passes you want per input size.
- Results will be saved in:
  ```
  benchmarks/results_all.csv
  ```

---

## Plot the Results

Make sure Python and required libraries are installed:

```bash
pip install pandas matplotlib seaborn
```

### Boxplot of all timings:

```bash
python plot_float4_compare.py
```

### Mean of best 10% (trimmed):

```bash
python plot_float4_compare_avg.py
```

### Output files:

- `benchmarks/exec_time_float4.png`
- `benchmarks/speedup_float4.png`
- `benchmarks/exec_time_best10.png`
- `benchmarks/speedup_best10.png`

---

## Notes

- Use `Release` builds for accurate timing results.
- Ensure all input sizes are divisible by 4 (required for `float4` kernels).
- Float4 kernels are not validated for correctness — only performance.
- This project is intended for machines with CUDA-capable GPUs.

---