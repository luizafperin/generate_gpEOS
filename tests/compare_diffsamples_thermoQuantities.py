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

def plot_eos_comparison(l, sigma, data_dir="/home/luizaperin/masters_research/gp/data_files/constrained/l10_s15/Testing_files/newStepsize"):
    sample_indices = [18, 20, 21]  # replace with your sample indices
    colors = ["tab:red", "tab:purple", "tab:orange",]  # add more colors if needed
    labels = [r"$s=18$", r"$s=20$", r"$s=21$"]  # replace with your labels

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
    plt.ylabel(r"$cs2$")
    plt.title(f"cs2 vs Temperature (l={l}, σ={sigma})")

    plt.grid(True, alpha=0.3)
    plt.legend()

    # set limits differently according to which thermo quantity is being plotted
    plt.xlim(0.03, 0.7)
    #plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig("/home/luizaperin/masters_research/hydro_res_analysis/plots_const/l10_s15/w/cs2_plots_s18_21.png")
    plt.show()

if __name__ == "__main__":
    # Example parameters
    l = 10
    sigma = 15
    plot_eos_comparison(l, sigma)
