import numpy as np
import matplotlib.pyplot as plt
import os

def load_eos_file(filename):
    """Load an EOS file and return arrays for cs2 and T."""
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip first line (length)
    data = np.loadtxt(lines)
    e, T, p, cs2, s = data.T
    return s, T


def plot_cs2_vs_T_jk():
    data_dir = "/home/luizaperin/masters_research/gp/data_files/jk_samples"

    sample_indices = [1, 2]
    colors = ["tab:blue", "tab:red"]

    plt.figure(figsize=(8, 6))

    for sample_idx, color in zip(sample_indices, colors):

        file_path = os.path.join(
            data_dir,
            f"eos_jk_run1_s{sample_idx}.dat"
        )

        print(file_path)

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        s, T = load_eos_file(file_path)

        plt.plot(
            T,
            s,
            color=color,
            linewidth=1,
            label=fr"$s={sample_idx}$"
        )

    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$s$ [GeV/fm$^3$]")
    plt.title(r"$s$ vs Temperature")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "/home/luizaperin/masters_research/hydro_res_analysis_run3/"
        "res_jk/s_vs_T_jk.png"
    )
    plt.show()


if __name__ == "__main__":
    plot_cs2_vs_T_jk()