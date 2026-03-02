# Build Instructions

This project benchmarks GPU kernel variants using CUDA. It supports building and running on both **Windows** and **Linux** systems.

## Requirements

- **CUDA Toolkit** (version 11.0 or newer)
- **CMake** (version 3.18+)
- **Python 3.8+** with:
  - `pandas`
  - `matplotlib`
  - `seaborn`
- C++ compiler with **C++20** support:
  - Windows: Visual Studio 2022 (with CUDA integration)
  - Linux: `g++` or `clang++`
- **Doxygen** (optional, for generating API documentation)

---

## Build (Windows)

> Recommended: Open "x64 Native Tools Command Prompt for VS 2022"

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

To build the debug version:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
```

---

## Build (Linux)

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

To build the debug version:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

---

## Executables

After building, you will get two binaries:

- `gpu_kernel_benchmark` – runs a single benchmark with generated input or from file
- `gpu_kernel_batch` – runs many passes across input sizes and saves results

---

## Run Examples

### Single Benchmark Run

Generate input file and run with all kernel variants:

```bash
./gpu_kernel_benchmark sqrt_log input_file
```

Run with specific variant and custom parameters:

```bash
./gpu_kernel_benchmark add input_file --variant global --warmup 20 --passes 500 --mode kernel
```

Command line options:
- `--variant <global|shared|float4|all>` - Select kernel variant (default: all)
- `--warmup <N>` - Number of warm-up launches (default: 20)
- `--passes <N>` - Number of timed measurement passes (default: 500)
- `--mode <kernel|e2e>` - Measurement mode: kernel-only or end-to-end (default: kernel)

### Batch Benchmark Run

Run batch benchmark with 100 passes per input size:

```bash
./gpu_kernel_batch sqrt_log 100
```

Run with specific variant and custom parameters:

```bash
./gpu_kernel_batch sqrt_log 100 --variant all --warmup 20 --mode kernel
```

> The second argument is the number of passes per input size.

Results are saved to `benchmarks/result.csv`

---

## Generate Documentation

This project includes professional Doxygen documentation for all public APIs, kernel launchers, and helpers.

Generate HTML documentation:

```bash
doxygen Doxyfile
```

Open `docs/index.html` in your browser to view the API documentation.

All functions are documented with:
- Function purpose and behavior
- Parameter descriptions
- Return value information
- Usage examples and requirements

---

## Python Plots

Install required packages:

```bash
pip install pandas matplotlib seaborn
```

Run plots:

```bash
python plot_float4_compare.py
python plot_float4_compare_avg.py
```

Generated plots will be saved in the `benchmarks/` directory.

---

## Troubleshooting

### CUDA Not Found

Ensure CUDA Toolkit is installed and the `CUDA_PATH` environment variable is set.

### Build Errors

- Verify C++20 compiler support
- Check CMake version (3.18+ required)
- Ensure CUDA compute capability matches your GPU

### Runtime Errors

- Verify CUDA drivers are up to date
- Check GPU memory availability
- Ensure input file format is correct (binary with int size + float arrays)

---

Author: Kamil Jatkowski, 2025
