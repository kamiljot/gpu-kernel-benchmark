## Project status

This project is actively maintained again. The previous archive notice is no longer valid. Current focus: stabilization, CLI validation, and benchmark quality improvements.
 
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
- Persistent kernel launchers that execute kernels internally multiple times to reduce CPU-GPU call overhead and improve measurement accuracy
- Support for selecting kernel variant via `--variant` CLI flag (e.g., `--variant global`, `shared`, `float4`, or `all`)
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
- `docs/` — Generated Doxygen documentation (HTML)
- Build scripts and docs (`CMakeLists.txt`, `BUILD.md`, `README.md`)

## Build Instructions

See [BUILD.md](./BUILD.md) for full setup. Minimal steps:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

**Note:** The default CUDA architecture is set to 75 (Turing). To build for your specific GPU, set:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=XX
```
where `XX` is your GPU's compute capability (e.g., 75 for RTX 20xx, 86 for RTX 30xx, 89 for RTX 40xx).

## Run Tests

After building, run the correctness tests:

```bash
ctest --output-on-failure
# or run individual tests:
./bin/test_add_correctness
./bin/test_sqrt_log_correctness
./bin/test_sin_cos_pow_relu_correctness
```

## Run Benchmark (Single Pass)

```bash
./gpu_kernel_benchmark <operation> <input_file> [--variant <global|shared|float4|all>] [--warmup <N>] [--passes <N>] [--mode <kernel|e2e>]
```

- If `<input_file>` does not exist, it will be auto-generated.  
- Example:

```bash
./gpu_kernel_benchmark add input_file --variant all --warmup 20 --passes 500 --mode kernel
```

## Run Benchmark (Batch Mode)

```bash
./gpu_kernel_batch <operation> <passes> [--variant <global|shared|float4|all>] [--warmup <N>] [--mode <kernel|e2e>]
```

- Example: run `100` passes for `sqrt_log`:

```bash
./gpu_kernel_batch sqrt_log 100 --variant all --warmup 20 --mode kernel
```

Results saved to `benchmarks/result.csv`

## Documentation

Generate Doxygen documentation (requires Doxygen installed):

```bash
doxygen Doxyfile
```

Open `docs/html/index.html` in your browser to view the generated API documentation.

All public functions, kernel launchers, and helpers are fully documented with Doxygen comments.

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
- JSON/HTML report generation  

---

Author: Kamil Jatkowski, 2025
