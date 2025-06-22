# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV (with multiple passes)
df = pd.read_csv("benchmarks/results_multi.csv")

# Boxplot: CPU times
plt.figure(figsize=(10, 5))
sns.boxplot(x="size", y="cpu_time_ms", data=df)
plt.title("CPU execution time (boxplot)")
plt.ylabel("Time [ms]")
plt.xlabel("Input size")
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmarks/cpu_boxplot.png")
plt.show()

# Boxplot: GPU times
plt.figure(figsize=(10, 5))
sns.boxplot(x="size", y="gpu_time_ms", data=df)
plt.title("GPU execution time (boxplot)")
plt.ylabel("Time [ms]")
plt.xlabel("Input size")
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmarks/gpu_boxplot.png")
plt.show()

# Boxplot: Speedup
df["speedup"] = df["cpu_time_ms"] / df["gpu_time_ms"]
plt.figure(figsize=(10, 5))
sns.boxplot(x="size", y="speedup", data=df)
plt.title("Speedup (CPU / GPU) - boxplot")
plt.ylabel("Speedup x")
plt.xlabel("Input size")
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmarks/speedup_boxplot.png")
plt.show()