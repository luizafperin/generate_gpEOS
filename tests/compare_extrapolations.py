import numpy as np
import matplotlib.pyplot as plt
import os

def load_eos_file(filename):
    """Load an EOS file and return arrays for e, T, p, cs2, and s."""
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip first line
    data = np.loadtxt(lines)
    return data.T  # e, T, p, cs2, s


def plot_eos(sample_idx, l, sigma, data_dir="./data_files/constrained/l10_s15"):
    labels = ["w", "hrg", "pwr"]
    colors = ["tab:blue", "tab:red", "tab:green"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for label, color in zip(labels, colors):

        # ===== Correct filename pattern =====
        file_path = os.path.join(
            data_dir,
            f"eos_{label}_s{sample_idx}_l{l}_s{sigma}.dat"
        )

        # ===== Check existence =====
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        # ===== Load data =====
        e, T, p, cs2, s = load_eos_file(file_path)

        # ===== Plot =====
        axs[0].plot(T, e, label=label, lw = 0.7, color=color)
        axs[1].plot(T, p, label=label, lw = 0.7, color=color)
        axs[2].plot(T, cs2, label=label, lw= 0.7, color=color)
        axs[3].plot(T, s, label=label, lw = 0.7, color=color)

    # ===== Labels =====
    axs[0].set_ylabel(r"$\varepsilon$ [GeV/fm$^3$]")
    axs[1].set_ylabel(r"$p$ [GeV/fm$^3$]")
    axs[2].set_ylabel(r"$c_s^2$")
    axs[3].set_ylabel(r"$s$ [1/fm$^3$]")

    axs[2].set_xlabel(r"$T$ [GeV]")
    axs[3].set_xlabel(r"$T$ [GeV]")

    # ===== Grid, legends, and uniform x-limits =====
    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0.08, 0.16)   # same limit for all plots

    # ===== Title & save =====
    fig.suptitle(f"Equation of State Comparison (sample {sample_idx}, l={l}, sigma={sigma})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/home/luizaperin/masters_research/hydro_res_analysis_run3/eos_file_plots_closer.png")
    plt.show()


if __name__ == "__main__":
    sample_idx = 0
    l = 10
    sigma = 15
    plot_eos(sample_idx, l, sigma)