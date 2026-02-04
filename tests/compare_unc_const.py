import numpy as np
import matplotlib.pyplot as plt
import os

def load_eos_file(filename):
    """Load an EOS file and return arrays for e, T, p, cs2, s."""
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip header
    data = np.loadtxt(lines)
    return data.T  # e, T, p, cs2, s


def plot_energy_multi_dirs(sample_idx, l, sigma,
                           dirs=["./data_files/constrained/l10_s15",
                                 "./data_files/unconstrained/l10_s15"]):

    labels = ["w", "hrg", "pwr"]
    colors = ["tab:blue", "tab:red", "tab:green"]
    styles = ["-", "--"]   # solid for constrained, dashed for unconstrained

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.flatten()

    ylabels = [
        r"$\varepsilon$ [GeV/fm$^3$]",
        r"$p$ [GeV/fm$^3$]",
        r"$c_s^2$",
        r"$s$ [1/fm$^3$]"
    ]

    for data_dir, style in zip(dirs, styles):
        dir_name = os.path.basename(data_dir)  # “constrained” or “unconstrained”

        for label, color in zip(labels, colors):

            file_path = os.path.join(
                data_dir,
                f"eos_{label}_s{sample_idx}_l{l}_s{sigma}.dat"
            )

            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue

            e, T, p, cs2, s = load_eos_file(file_path)

            axs[0].plot(T, e, lw = 0.7, label=f"{label}",
                        color=color, linestyle=style, linewidth=2)

            axs[1].plot(T, p, lw = 0.7, label=f"{label}",
                        color=color, linestyle=style, linewidth=2)

            axs[2].plot(T, cs2, lw = 0.7,  label=f"{label}",
                        color=color, linestyle=style, linewidth=2)

            axs[3].plot(T, s, lw = 0.7, label=f"{label}",
                        color=color, linestyle=style, linewidth=2)

    # Formatting
    for ax, ylabel in zip(axs, ylabels):
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"$T$ [GeV]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"EOS Comparison (sample {sample_idx}, l={l}, σ={sigma})",
                 fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("/home/luizaperin/masters_research/hydro_res_analysis_run3/eos_comparison_multi_dirs.png")
    plt.show()


if __name__ == "__main__":
    sample_idx = 1
    l = 10
    sigma = 15
    plot_energy_multi_dirs(sample_idx, l, sigma)
