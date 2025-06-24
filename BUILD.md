
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

Generate input file and run:

```bash
./gpu_kernel_benchmark sqrt_log input_file
```

Run batch benchmark:

```bash
./gpu_kernel_batch sqrt_log 100
```

> The second argument is the number of passes per input size.

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
