import numpy as np
import matplotlib.pyplot as plt
import os

def load_eos_file(filename):
    """Load an EOS file and return arrays for e, T, p, cs2, and s."""
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip the first line with the length
    data = np.loadtxt(lines)
    e, T, p, cs2, s = data.T
    return e, T, p, cs2, s

def plot_eos_comparison(l, sigma, data_dir="/home/luizaperin/masters_research/gp/data_files/constrained/l10_s15"):
    sample_indices = [0, 1, 2]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = [r"$s=0$", r"$s=1$", r"$s=2$"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for sample_idx, color, label in zip(sample_indices, colors, labels):
        file_path = os.path.join(data_dir, f"eos_w_s{sample_idx}_l{l}_s{sigma}.dat")
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        e, T, p, cs2, s = load_eos_file(file_path)

        axs[0].plot(T, e, label=label, lw= 0.7, color=color)
        axs[0].set_ylabel(r"$\varepsilon$ [GeV/fm$^3$]")

        axs[1].plot(T, p, label=label, lw= 0.7, color=color)
        axs[1].set_ylabel(r"$p$ [GeV/fm$^3$]")

        axs[2].plot(T, cs2, label=label, lw= 0.7, color=color)
        axs[2].set_ylabel(r"$c_s^2$")
        axs[2].set_xlabel(r"$T$ [GeV]")

        axs[3].plot(T, s, label=label,lw= 0.7, color=color)
        axs[3].set_ylabel(r"$s$ [1/fm$^3$]")
        axs[3].set_xlabel(r"$T$ [GeV]")

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel(r"$T$ [GeV]")
        #ax.set_xlim(0, 0.2)

    fig.suptitle(f"Equation of State Comparison (l={l}, sigma={sigma})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/home/luizaperin/masters_research/hydro_res_analysis_run3/res_pwr_unc/l15_s30/eos_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Example parameters
    l = 15
    sigma = 30
    plot_eos_comparison(l, sigma)
