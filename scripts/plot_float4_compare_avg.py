"""
plot_float4_compare_avg.py
--------------------------
Plots average (mean of best 10%) execution times and speedups for CPU and GPU kernel benchmarks.

- Loads results from CSV
- Filters for best 10% times per input size (trimming slow outliers)
- Plots log-log execution time and linear speedup charts

Author: Kamil J.
Date: 2025-07-07
"""

import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    Loads results, filters best 10%, and generates execution time / speedup plots.
    """
    # Load benchmark results — column names match benchmark_utils.cpp output
    df = pd.read_csv("benchmarks/result.csv")

    # Compute speedups
    df["speedup_global"] = df["cpu_ms"] / df["gpu_global_ms"]
    df["speedup_shared"] = df["cpu_ms"] / df["gpu_shared_ms"]
    df["speedup_float4"] = df["cpu_ms"] / df["gpu_float4_ms"]

    def keep_best_10_percent(series):
        """
        Returns only the best 10% (lowest) values from a pandas Series.
        """
        cutoff = series.quantile(0.10)
        return series[series <= cutoff]

    def avg_best_10(dataframe, column):
        """
        Calculates the mean of the best 10% values for each input size.
        """
        return dataframe.groupby("N")[column].apply(keep_best_10_percent).groupby("N").mean()

    sizes = sorted(df["N"].unique())

    # === 1. Average execution times (log scale) ===
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, avg_best_10(df, "cpu_ms"), label="CPU", marker='o')
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


if __name__ == "__main__":
    main()