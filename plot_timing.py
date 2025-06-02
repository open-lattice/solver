import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

# --- Setup ---
sns.set_theme(style="whitegrid")
df = pd.read_csv("timing_log.csv")

# --- Type corrections and cleanup ---
numeric_cols = ["rows", "cols", "nnz", "sparsity", "constraints", "total_time_s",
                "constraint_time_s", "procs", "speedup", "efficiency"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=numeric_cols)  # Drop rows with bad data

df["matrix"] = df["matrix"].apply(lambda x: os.path.basename(str(x)).replace(".bin", "").replace("./binaries/", ""))
df["procs"] = df["procs"].astype(int)

df["ratio"] = df["constraint_time_s"] / df["total_time_s"]

# --- Output folder ---
plot_folder = "Plots"
os.makedirs(plot_folder, exist_ok=True)

# --- 1. 3D Speedup Plot ---
for matrix in df["matrix"].unique():
    df_matrix = df[df["matrix"] == matrix]

    pivot = df_matrix.pivot_table(index="constraints", columns="procs", values="speedup", aggfunc="mean")
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("MPI Processes")
    ax.set_ylabel("Constraints")
    ax.set_zlabel("Speedup")
    ax.set_title(f"3D Speedup for {matrix}")
    ax.view_init(elev=30, azim=135)
    fig.colorbar(surf, shrink=0.6, aspect=10, label="Speedup")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"3D_speedup_{matrix}.png"))
    plt.close()

# --- 2. 2D Line Plots: Total Time vs Procs | Constraint Time vs Constraints ---
for matrix in df["matrix"].unique():
    df_matrix = df[df["matrix"] == matrix]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Timing Breakdown: {matrix}", fontsize=15)

    # Total Time vs MPI Processes
    sns.lineplot(data=df_matrix, x="procs", y="total_time_s", hue="constraints", marker="o", ax=axs[0])
    axs[0].set_title("Total Time vs MPI Processes")
    axs[0].set_xlabel("MPI Processes")
    axs[0].set_ylabel("Total Time (s)")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Constraint Time vs Constraints (with filtering)
    for proc in sorted(df_matrix["procs"].unique()):
        df_proc = df_matrix[df_matrix["procs"] == proc]
        if len(df_proc["constraints"].unique()) > 1:
            axs[1].plot(df_proc["constraints"], df_proc["constraint_time_s"], marker="o", label=f"{proc} procs")
    axs[1].set_title("Constraint Time vs Constraints")
    axs[1].set_xlabel("Number of Constraints")
    axs[1].set_ylabel("Constraint Time (s)")
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].legend(title="Processes")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plot_folder, f"2D_timing_{matrix}.png"))
    plt.close()

# --- 3. Bar Plot: Constraint Time / Total Time ratio ---
avg_ratio = df.groupby("procs")["ratio"].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(data=avg_ratio, x="procs", y="ratio", palette="Blues_d")
plt.title("Average Constraint-to-Total Time Ratio by MPI Processes")
plt.xlabel("MPI Processes")
plt.ylabel("Avg. Constraint Time / Total Time")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "bar_constraint_ratio.png"))
plt.close()

# --- 4. Matrix Summary Table ---
summary_cols = ["matrix", "rows", "cols", "nnz", "sparsity"]
summary_df = df[summary_cols].drop_duplicates().sort_values(by="matrix")
summary_df.to_csv("matrix_summary_table.csv", index=False)

print("\nMatrix Summary Table:\n")
print(tabulate(summary_df, headers='keys', tablefmt='github', showindex=False))

# --- 5. 2D Efficiency vs Process Number (per constraint) ---
for matrix in df["matrix"].unique():
    df_matrix = df[df["matrix"] == matrix]
    fig, ax = plt.subplots(figsize=(10, 6))

    for constraint in sorted(df_matrix["constraints"].unique()):
        df_constraint = df_matrix[df_matrix["constraints"] == constraint]
        if len(df_constraint["procs"].unique()) > 1:
            ax.plot(df_constraint["procs"], df_constraint["efficiency"], marker="o", label=f"{constraint} constraints")

    ax.set_title(f"Efficiency vs MPI Processes: {matrix}")
    ax.set_xlabel("MPI Processes")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0, 1.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Constraints")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"2D_efficiency_{matrix}.png"))
    plt.close()
