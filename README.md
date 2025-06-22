# GPU Kernel Benchmark

This project benchmarks the performance of the operation:

```cpp
c[i] = sqrtf(a[i]) + logf(b[i])
```

on CPU vs CUDA GPU, across multiple vector sizes.

## Build Instructions
```bash
mkdir build && cd build
cmake ..
make
```

## Usage
```bash
./gpu_kernel_benchmark
```

## Output
Results are written to `benchmarks/results_multi.csv` in the format:
```
size,pass,cpu_time_ms,gpu_time_ms
```

Each line represents one measurement pass for a given input size.

## TODO
- [x] Measure kernel execution time (excluding memcpy)
- [x] Benchmark for multiple input sizes
- [x] Multiple passes for boxplot analysis
- [ ] Shared memory variant
- [ ] Export plot
- [ ] SYCL version for comparison
