import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("benchmarks/results.csv")

# Plot CPU vs GPU time
plt.figure(figsize=(8, 5))
plt.plot(df["size"], df["cpu_time_ms"], label="CPU", marker="o")
plt.plot(df["size"], df["gpu_time_ms"], label="GPU (kernel only)", marker="o")
plt.xlabel("Input size (number of floats)")
plt.ylabel("Time [ms]")
plt.title("Execution Time: CPU vs GPU")
plt.legend()
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.savefig("cpu_vs_gpu_time.png")
plt.show()

# Plot speedup
speedup = df["cpu_time_ms"] / df["gpu_time_ms"]
plt.figure(figsize=(8, 5))
plt.plot(df["size"], speedup, marker="o", color="green")
plt.xlabel("Input size (number of floats)")
plt.ylabel("Speedup (CPU / GPU)")
plt.title("Speedup achieved by GPU")
plt.grid(True)
plt.xscale("log")
plt.tight_layout()
plt.savefig("gpu_speedup.png")
plt.show()

import os
print("Saved files in:", os.getcwd())