import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_eos_file(filename):
    """Load an EOS file and return arrays for e, T, p, cs2, and s."""
    with open(f"./data_files/constrained/eos_w_s{sample_idx}_l{l}_s_{sigma}.dat") as f:
        lines = f.readlines()[1:]  # skip the first line with the length
    data = np.loadtxt(lines)
    e, T, p, cs2, s = data.T
    return e, T, p, cs2, s

def plot_eos(sample_idx, l, sigma, data_dir="./data_files/constrained"):
    labels = ["w", "hrg", "pwr"]
    colors = ["tab:blue"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for label, color in zip(labels, colors):
        file_path = os.path.join(data_dir, f"eos_w_s{sample_idx}_l{l}_s_{sigma}.dat")
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        e, T, p, cs2, s = load_eos_file(file_path)

        axs[0].plot(T, e, label=label, color=color)
        axs[0].set_ylabel(r"$\varepsilon$ [GeV/fm$^3$]")

        axs[1].plot(T, p, label=label, color=color)
        axs[1].set_ylabel(r"$p$ [GeV/fm$^3$]")

        axs[2].plot(T, cs2, label=label, color=color)
        axs[2].set_ylabel(r"$c_s^2$")
        axs[2].set_xlabel(r"$T$ [GeV]")

        axs[3].plot(T, s, label=label, color=color)
        axs[3].set_ylabel(r"$s$ [1/fm$^3$]")
        axs[3].set_xlabel(r"$T$ [GeV]")

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel(r"$T$ [GeV]")

    fig.suptitle(f"Equation of State (sample {sample_idx}, l={l}, sigma={sigma})", fontsize=14)
    plt.tight_layout()
    plt.savefig("eos_file_plots.png")
    plt.show()


if __name__ == "__main__":
    # Example: you can change these parameters
    sample_idx = 2
    l = 10
    sigma = 15
    plot_eos(sample_idx, l, sigma)
