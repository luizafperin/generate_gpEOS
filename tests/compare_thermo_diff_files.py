import numpy as np
import matplotlib.pyplot as plt
import os

def load_eos_file(filename):
    """Load an EOS file and return arrays for e and T."""
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip the first line with the length
    data = np.loadtxt(lines)
    e, T, p, cs2, s = data.T
    return cs2, T

def plot_e_vs_T_for_eostypes(sample_idx=1, l=10, sigma=15):
    data_dir=f"/home/luizaperin/masters_research/gp/data_files/constrained/l{l}_s{sigma}"
    """Compare e vs T for different EOS types (e.g., HRG vs HotQCD) for the same sample."""
    eos_types = ["hrg", "w", "pwr"]  # you can add more types if needed
    colors = ["tab:blue", "tab:red", "tab:green"]
    labels = [r"HRG", r"Weighted", r"Pwr"]

    plt.figure(figsize=(8, 6))

    for eos_type, color, label in zip(eos_types, colors, labels):
        file_path = os.path.join(data_dir, f"eos_{eos_type}_s{sample_idx}_l{l}_s{sigma}.dat")
        print(file_path)
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        e, T = load_eos_file(file_path)
        plt.plot(T, e, label=label, color=color, linewidth=1)

    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$ cs2 $")
    plt.title(f"cs2 vs Temperature (s={sample_idx}, l={l}, σ={sigma})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/home/luizaperin/masters_research/hydro_res_analysis_run3/plots_const/l{l}_s{sigma}/cs2_vs_T_s{sample_idx}_comparison_unc.png")
    plt.show()

if __name__ == "__main__":
    # Example usage
    plot_e_vs_T_for_eostypes(sample_idx=23, l=10, sigma=15)
