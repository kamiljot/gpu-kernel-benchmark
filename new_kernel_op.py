import os
import sys
import re

HEADER = """
// Host launchers for {name} kernel variants (global, shared memory, float4).

#pragma once

// Runs the global memory {name} kernel.
// N: number of elements.
extern "C" float run_{name}_global(const float* a, const float* b, float* c, int N);

// Runs the shared memory {name} kernel.
// N: number of elements.
extern "C" float run_{name}_shared(const float* a, const float* b, float* c, int N);

// Runs the float4 vectorized {name} kernel.
// N: number of elements.
extern "C" float run_{name}_float4(const float* a, const float* b, float* c, int N);
"""

CUH = """
// Device kernel declarations for {name} kernel (global, shared, float4).

#pragma once
#include <cuda_runtime.h>

// Global memory kernel.
__global__ void {name}_global_kernel(const float* a, const float* b, float* c, int N);

// Shared memory kernel.
__global__ void {name}_shared_kernel(const float* a, const float* b, float* c, int N);

// float4 vectorized kernel.
__global__ void {name}_float4_kernel(const float4* a, const float4* b, float4* c, int N);
"""

CU = """
// CUDA kernel implementations for {name} (global, shared, float4).

#include "{name}_kernels.cuh"

// TODO: Implement global memory kernel for {name}
__global__ void {name}_global_kernel(const float* a, const float* b, float* c, int N) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {{
        // TODO: Implement computation for {name} (global memory)
        c[idx] = 0.0f;
    }}
}}

// TODO: Implement shared memory kernel for {name}
__global__ void {name}_shared_kernel(const float* a, const float* b, float* c, int N) {{
    extern __shared__ float shmem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Implement shared memory usage for {name}
    if (idx < N) {{
        c[idx] = 0.0f;
    }}
}}

// TODO: Implement float4 kernel for {name}
__global__ void {name}_float4_kernel(const float4* a, const float4* b, float4* c, int N) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Each thread processes one float4 (4 elements)
    if (idx < N) {{
        c[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }}
}}
"""

LAUNCHER = """
// Implements host launchers for all {name} kernel variants.

#include "{name}.h"
#include "{name}_kernels.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

// Runs the global memory {name} kernel and measures execution time.
extern "C" float run_{name}_global(const float* a, const float* b, float* c, int N) {{
    // TODO: Implement host launcher logic for global kernel (allocate/copy, launch, sync, timing)
    throw std::runtime_error("run_{name}_global: Not implemented");
    return 0.0f;
}}

// Runs the shared memory {name} kernel and measures execution time.
extern "C" float run_{name}_shared(const float* a, const float* b, float* c, int N) {{
    // TODO: Implement host launcher logic for shared kernel (allocate/copy, launch, sync, timing)
    throw std::runtime_error("run_{name}_shared: Not implemented");
    return 0.0f;
}}

// Runs the float4 vectorized {name} kernel and measures execution time.
extern "C" float run_{name}_float4(const float* a, const float* b, float* c, int N) {{
    // TODO: Implement host launcher logic for float4 kernel (allocate/copy, launch, sync, timing)
    throw std::runtime_error("run_{name}_float4: Not implemented");
    return 0.0f;
}}
"""

CPU_BASELINE_DECL = "float run_cpu_{name}(const float* a, const float* b, float* c, int N);\n"

CPU_BASELINE_IMPL = """
float run_cpu_{name}(const float* a, const float* b, float* c, int N) {{
    // TODO: Implement CPU reference version of {name}
    return 0.0f;
}}
"""

def update_cpu_baseline_h(kernel_name):
    path = os.path.join("include", "cpu_baseline.h")
    decl = f"float run_cpu_{kernel_name}(const float* a, const float* b, float* c, int N);\n"
    if not os.path.exists(path):
        print(f"{path} not found, skipping cpu_baseline.h update.")
        return
    with open(path, "r") as f:
        lines = f.readlines()
    if any(decl.strip() in line for line in lines):
        print(f"Declaration for {kernel_name} already present in cpu_baseline.h")
        return
    # Add declaration at the end of the file
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    lines.append(decl)
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"Added declaration for {kernel_name} to cpu_baseline.h")

def update_cpu_baseline_cpp(kernel_name):
    path = os.path.join("src", "cpu_baseline.cpp")
    impl = CPU_BASELINE_IMPL.format(name=kernel_name)
    if not os.path.exists(path):
        print(f"{path} not found, skipping cpu_baseline.cpp update.")
        return
    with open(path, "r") as f:
        content = f.read()
    if f"run_cpu_{kernel_name}" in content:
        print(f"Function run_cpu_{kernel_name} already present in cpu_baseline.cpp")
        return
    with open(path, "a") as f:
        f.write(impl)
    print(f"Added function stub for {kernel_name} to cpu_baseline.cpp")

def update_kernel_dispatch(kernel_name):
    kd_path = os.path.join("src", "kernel_dispatch.cpp")
    include_line = f'#include "kernels/{kernel_name}/{kernel_name}.h"\n'

    new_condition = f'''
    else if (operation == "{kernel_name}") {{
        result.cpu_time = run_cpu_{kernel_name}(a, b, c, N);
        result.gpu_global_time = run_{kernel_name}_global(a, b, c, N);
        result.gpu_shared_time = run_{kernel_name}_shared(a, b, c, N);
        result.gpu_float4_time = run_{kernel_name}_float4(a, b, c, N);
    }}'''

    if not os.path.exists(kd_path):
        print(f"Warning: {kd_path} not found, skipping kernel_dispatch update.")
        return

    with open(kd_path, "r") as f:
        content = f.read()

    # Add include if missing
    if include_line.strip() not in content:
        includes = re.findall(r'#include .+\n', content)
        if includes:
            last_include = includes[-1]
            content = content.replace(last_include, last_include + include_line)
        else:
            content = include_line + content

    # Insert new_condition before the final else with throw
    pattern = r'(if\s*\(operation == "[^"]+"\)[\s\S]+?)(else\s*{\s*throw std::invalid_argument[^\}]+\})'
    match = re.search(pattern, content)
    if not match:
        print("Could not find dispatch if-else block")
        return

    before_else = match.group(1)
    else_block = match.group(2)

    if kernel_name in before_else:
        print(f"Kernel {kernel_name} already in dispatch")
        return

    new_before_else = before_else + new_condition + "\n"
    new_content = content.replace(before_else + else_block, new_before_else + else_block)

    with open(kd_path, "w") as f:
        f.write(new_content)

    print(f"Updated kernel_dispatch.cpp with {kernel_name} condition correctly.")

def update_readme(kernel_name):
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        print(f"Warning: {readme_path} not found, skipping README update.")
        return

    with open(readme_path, "r") as f:
        lines = f.readlines()

    # Update operations list in Features section
    for i, line in enumerate(lines):
        if line.strip().startswith("- Support for different operations:"):
            if f"`{kernel_name}`" not in line:
                # Insert kernel_name before (more to come)
                if "(more to come)" in line:
                    new_line = line.replace(
                        "(more to come)",
                        f"`{kernel_name}`, (more to come)"
                    )
                else:
                    # If no "more to come", just append
                    new_line = line.rstrip() + f", `{kernel_name}`\n"
                lines[i] = new_line
            break

    # Add usage example to How to add a new kernel operator section
    usage_line = f"   ```bash\n   python3 new_kernel_op.py {kernel_name}\n   ```\n"
    section_start = None
    for i, line in enumerate(lines):
        if "How to add a new kernel operator" in line:
            section_start = i
            break

    if section_start is not None:
        # Check if usage_line already present
        usage_present = any(usage_line.strip() in l for l in lines)
        if not usage_present:
            # Insert usage example 2 lines below section header
            insert_pos = section_start + 2
            if insert_pos < len(lines):
                lines.insert(insert_pos, usage_line)
            else:
                lines.append(usage_line)

    with open(readme_path, "w") as f:
        f.writelines(lines)

    print(f"Updated README.md with kernel {kernel_name}.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 new_kernel_op.py <kernel_name>")
        sys.exit(1)

    kernel = sys.argv[1].lower()
    kernel_dir = os.path.join("src", "kernels", kernel)
    os.makedirs(kernel_dir, exist_ok=True)

    files = {
        f"{kernel}.h": HEADER.format(name=kernel),
        f"{kernel}_kernels.cuh": CUH.format(name=kernel),
        f"{kernel}_kernels.cu": CU.format(name=kernel),
        f"{kernel}_launcher.cu": LAUNCHER.format(name=kernel),
    }

    for fname, content in files.items():
        path = os.path.join(kernel_dir, fname)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(content)
            print(f"Created: {path}")
        else:
            print(f"Exists, skipping: {path}")

    update_kernel_dispatch(kernel)
    update_cpu_baseline_h(kernel)
    update_cpu_baseline_cpp(kernel)
    update_readme(kernel)

if __name__ == "__main__":
    main()