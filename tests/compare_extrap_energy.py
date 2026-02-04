import numpy as np
import matplotlib.pyplot as plt
import os

def load_eos_file(filename):
    """Load an EOS file and return arrays for e, T, p, cs2, and s."""
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip header line
    data = np.loadtxt(lines)
    return data.T  # returns e, T, p, cs2, s


def plot_energy(sample_idx, l, sigma, data_dir="./data_files/constrained/l10_s15"):
    labels = ["w", "hrg", "pwr"]
    colors = ["tab:blue", "tab:red", "tab:green"]

    plt.figure(figsize=(8, 6))

    for label, color in zip(labels, colors):

        # File path for each label
        file_path = os.path.join(
            data_dir,
            f"eos_{label}_s{sample_idx}_l{l}_s{sigma}.dat"
        )

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        e, T, p, cs2, s = load_eos_file(file_path)

        plt.plot(T, e, label=label, color=color)

    # ---- Labels and formatting ----
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$\varepsilon$ [GeV/fm$^3$]")
    plt.title(f"Energy vs Temperature (sample {sample_idx}, l={l}, σ={sigma})")

    plt.grid(True, alpha=0.3)
    plt.legend()

    # Same x-range for all
    plt.xlim(0, 0.2)

    plt.tight_layout()
    plt.savefig("/home/luizaperin/masters_research/hydro_res_analysis_run3/energy_file_plots_closer.png")
    plt.show()


if __name__ == "__main__":
    sample_idx = 0
    l = 10
    sigma = 15
    plot_energy(sample_idx, l, sigma)
