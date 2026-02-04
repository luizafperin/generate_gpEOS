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

def plot_eos_comparison(l, sigma, data_dir="/home/luizaperin/masters_research/gp/data_files/unconstrained/l15_s30"):
    sample_indices = [0, 1, 2]
    colors = ["tab:purple", "tab:blue", "tab:green"]
    labels = [r"$s=0$", r"$s=1$", r"$s=2$"]

    plt.figure(figsize=(10, 8))

    for sample_idx, color, label in zip(sample_indices, colors, labels):
        file_path = os.path.join(data_dir, f"eos_w_s{sample_idx}_l{l}_s{sigma}.dat")
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        e, T, p, cs2, s = load_eos_file(file_path)

        plt.plot(T, cs2, label=label, color=color)        

    # ---- Labels and formatting ----
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$s$ [1/fm$^3$]")
    plt.title(f"Entropy vs Temperature (l={l}, σ={sigma})")

    plt.grid(True, alpha=0.3)
    plt.legend()

    # set limits differently according to which thermo quantity is being plotted
    plt.xlim(0.060, 0.1)
    plt.ylim(0.1, 0.2)

    plt.tight_layout()
    plt.savefig("/home/luizaperin/masters_research/hydro_res_analysis_run3/plots_unc/l15_s30/w/entropy_file_plots_s1.png")
    plt.show()

if __name__ == "__main__":
    # Example parameters
    l = 15
    sigma = 30
    plot_eos_comparison(l, sigma)
