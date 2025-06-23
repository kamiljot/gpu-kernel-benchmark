import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark results
df = pd.read_csv("benchmarks/results_all.csv")

# Compute speedups
df["speedup_global"] = df["cpu_time_ms"] / df["gpu_global_ms"]
df["speedup_shared"] = df["cpu_time_ms"] / df["gpu_shared_ms"]
df["speedup_float4"] = df["cpu_time_ms"] / df["gpu_float4_ms"]

# === TRIMMING: only keep best 10% (lowest times) ===
def keep_best_10_percent(series):
    cutoff = series.quantile(0.10)
    return series[series <= cutoff]

# Average best 10% per input size
def avg_best_10(df, column):
    return df.groupby("size")[column].apply(keep_best_10_percent).groupby("size").mean()

# List of sizes
sizes = sorted(df["size"].unique())

# === 1. Average execution times (log scale) ===
plt.figure(figsize=(10, 5))
plt.plot(sizes, avg_best_10(df, "cpu_time_ms"), label="CPU", marker='o')
plt.plot(sizes, avg_best_10(df, "gpu_global_ms"), label="GPU Global", marker='o')
plt.plot(sizes, avg_best_10(df, "gpu_shared_ms"), label="GPU Shared", marker='o')
plt.plot(sizes, avg_best_10(df, "gpu_float4_ms"), label="GPU Float4", marker='o')
plt.title("Execution Time (Mean of Best 10%)")
plt.xlabel("Input Size")
plt.ylabel("Time [ms]")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("benchmarks/exec_time_best10.png")
plt.show()

# === 2. Average speedups (linear scale) ===
plt.figure(figsize=(10, 5))
plt.plot(sizes, avg_best_10(df, "speedup_global"), label="Global", marker='o')
plt.plot(sizes, avg_best_10(df, "speedup_shared"), label="Shared", marker='o')
plt.plot(sizes, avg_best_10(df, "speedup_float4"), label="Float4", marker='o')
plt.title("Speedup over CPU (Mean of Best 10%)")
plt.xlabel("Input Size")
plt.ylabel("Speedup x")
plt.xscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("benchmarks/speedup_best10.png")
plt.show()