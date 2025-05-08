import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

df = pd.read_csv("timing_log.csv")

df["matrix"] = df["matrix"].apply(lambda x: os.path.basename(str(x)).replace(".mtx", "").replace("./matrices/", ""))
df["procs"] = df["procs"].astype(int)

plot_folder = "Plots"

os.makedirs(plot_folder, exist_ok=True)

sns.set_theme(style="whitegrid")

for matrix in df["matrix"].unique():
    df_matrix = df[df["matrix"] == matrix]

    # -------- 3D Speedup Plot --------
    pivot = df_matrix.pivot_table(index="constraints", columns="procs", values="speedup", aggfunc="mean")
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", linewidth=0.5, antialiased=True)

    ax.set_xlabel("MPI Processes", labelpad=10)
    ax.set_ylabel("Constraints", labelpad=10)
    ax.set_zlabel("Speedup", labelpad=10)
    ax.set_title(f"3D Speedup for Matrix: {matrix}", pad=15)
    ax.view_init(elev=30, azim=135)
    ax.set_box_aspect([1.2, 1, 0.6])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=6))

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Speedup")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"3D_speedup_{matrix}.png"))
    plt.close()

    # -------- 2D Constraint Time Plots --------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Timing Analysis: {matrix}", fontsize=14)

    sns.lineplot(data=df_matrix, x="procs", y="constraint_time_s", hue="constraints", marker="o", ax=axs[0])
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_title("Constraint Time vs MPI Processes")
    axs[0].set_xlabel("MPI Processes")
    axs[0].set_ylabel("Constraint Time (s)")

    sns.lineplot(data=df_matrix, x="constraints", y="constraint_time_s", hue="procs", marker="o", ax=axs[1])
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_title("Constraint Time vs Constraints")
    axs[1].set_xlabel("Number of Constraints")
    axs[1].set_ylabel("Constraint Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plot_folder, f"2D_constraint_time_{matrix}.png"))
    plt.close()
