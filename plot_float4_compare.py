import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV results
df = pd.read_csv("benchmarks/results_all.csv")

# Compute speedup for each GPU variant
df["speedup_global"] = df["cpu_time_ms"] / df["gpu_global_ms"]
df["speedup_shared"] = df["cpu_time_ms"] / df["gpu_shared_ms"]
df["speedup_float4"] = df["cpu_time_ms"] / df["gpu_float4_ms"]

# Set plotting style
sns.set(style="whitegrid")

# === 1. Full execution time comparison (CPU + all GPU) ===
plt.figure(figsize=(12, 6))
df_melted = pd.melt(df, id_vars=["size"], value_vars=[
    "cpu_time_ms", "gpu_global_ms", "gpu_shared_ms", "gpu_float4_ms"
])
sns.boxplot(data=df_melted, x="size", y="value", hue="variable", showfliers=False)
plt.title("Execution Time: CPU vs GPU Kernels")
plt.xlabel("Input Size")
plt.ylabel("Time [ms]")
plt.legend(title="Variant")
plt.tight_layout()
plt.savefig("benchmarks/exec_time_float4.png")
plt.show()

# === 2. GPU-only comparison (global vs shared vs float4) ===
plt.figure(figsize=(12, 6))
df_gpu = pd.melt(df, id_vars=["size"], value_vars=[
    "gpu_global_ms", "gpu_shared_ms", "gpu_float4_ms"
])
sns.boxplot(data=df_gpu, x="size", y="value", hue="variable", showfliers=False)
plt.title("GPU Kernel Time: Global vs Shared vs Float4")
plt.xlabel("Input Size")
plt.ylabel("Time [ms]")
plt.legend(title="GPU Kernel")
plt.tight_layout()
plt.savefig("benchmarks/gpu_only_float4.png")
plt.show()

# === 3. Speedup over CPU (higher is better) ===
plt.figure(figsize=(12, 6))
df_speedup = pd.melt(df, id_vars=["size"], value_vars=[
    "speedup_global", "speedup_shared", "speedup_float4"
])
sns.boxplot(data=df_speedup, x="size", y="value", hue="variable", showfliers=False)
plt.title("Speedup vs CPU (log+sqrt benchmark)")
plt.xlabel("Input Size")
plt.ylabel("Speedup x")
plt.legend(title="GPU Kernel")
plt.tight_layout()
plt.savefig("benchmarks/speedup_float4.png")
plt.show()