# GPU Kernel Benchmark

This project benchmarks multiple GPU kernel implementations for common math operations on large arrays, comparing CPU performance against different GPU memory access strategies: global memory, shared memory, and float4 vectorized memory.

## Features

- Modular design with separate kernel, launcher, and utility files
- Support for different operations: `sqrt+log`, `add`, `sin_cos_pow_relu` (more to come)
- Easy addition of new kernels via automated code generation script
- Three GPU variants per operation:
  - Global memory access
  - Shared memory
  - float4 vectorization
- Automatic input data generation if missing
- CSV logging of benchmark results
- Python scripts for plotting results
- Optional batch mode benchmarking with configurable passes

## Project Structure (Overview)

- `include/` — Public header files  
- `src/` — Source files, including kernels and utilities  
- `src/kernels/` — Separate directories for each kernel operator  
- `benchmarks/` — Generated benchmark CSV files and plots  
- `scripts/` — Python plotting and utility scripts  
- Build scripts and docs (`CMakeLists.txt`, `BUILD.md`, `README.md`)

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

---

## How to add a new kernel operator

   ```bash
   python3 new_kernel_op.py sin_cos_pow_relu
   ```
This project is designed to be easily extensible by adding new GPU kernel operators.

### Steps to add a new kernel operator:

1. Run the provided script to generate boilerplate files:

   ```bash
   python3 new_kernel_op.py <kernel_name>
   ```

   Replace `<kernel_name>` with your desired operator name, e.g.:

   ```bash
   python3 new_kernel_op.py sin_cos_pow_relu
   ```

2. The script will create a new directory under `src/kernels/<kernel_name>/` containing the following files:

   - `<kernel_name>.h` — Host launcher declarations  
   - `<kernel_name>_kernels.cuh` — Device kernel declarations  
   - `<kernel_name>_kernels.cu` — CUDA kernel implementations with template functions  
   - `<kernel_name>_launcher.cu` — Host launcher implementations with function templates  

3. Implement your kernels and launcher functions inside the generated files.

4. The build system automatically detects all kernel source files under `src/kernels/` and includes them in the build. No need to manually edit build files.

5. Add support for your new operator in `kernel_dispatch.cpp` so it can be invoked via the CLI.

---

## Planned Improvements

The project roadmap includes:

- Adding more math operations (mul, sin+exp, etc.)  
- CLI support for individual kernel variant selection  
- JSON/HTML report generation  

---

Author: Kamil Jatkowski, 2025