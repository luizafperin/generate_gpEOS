import numpy as np
from scipy.interpolate import interp1d
import os

def load_eos_file(filename):
    """Load an EOS file and return arrays for e, T, p, cs2, and s."""
    with open(filename) as f:
        lines = f.readlines()[1:]  # skip the first line with the length
    data = np.loadtxt(lines)
    e, T, p, cs2, s = data.T
    return e, T, p, cs2, s

def evenly_spaced_e(sample_idx=3, l=10, sigma=15, data_dir="./data_files/unconstrained/"):
    eos_types = ["hrg", "w", "pwr"]

    # Define evenly spaced energy grid
    e_grid = np.arange(0, 500, 0.001)

    for eos_type in eos_types:
        file_path = os.path.join(data_dir, f"eos_{eos_type}_s{sample_idx}_l{l}_s_{sigma}.dat")

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        e, T, p, cs2, s = load_eos_file(file_path)

        # Create interpolators
        T_interp = interp1d(e, T, kind='linear', bounds_error=False, fill_value="extrapolate")
        p_interp = interp1d(e, p, kind='linear', bounds_error=False, fill_value="extrapolate")
        cs2_interp = interp1d(e, cs2, kind='linear', bounds_error=False, fill_value="extrapolate")
        s_interp = interp1d(e, s, kind='linear', bounds_error=False, fill_value="extrapolate")

        # Interpolated values on uniform e_grid
        T_even = T_interp(e_grid)
        p_even = p_interp(e_grid)
        cs2_even = cs2_interp(e_grid)
        s_even = s_interp(e_grid)

        output_file = f"./eos_{eos_type}_s{sample_idx}_l{l}_s{sigma}_even.dat"
        with open(output_file, "w") as f:
            f.write(f"{len(e_grid)}\n")
            for ei, Ti, pi, cs2i, si in zip(e_grid, T_even, p_even, cs2_even, s_even):
                f.write(f"{ei:.6f} {Ti:.6f} {pi:.6f} {cs2i:.6f} {si:.6f}\n")

        print(f"✅ Saved evenly spaced EOS: {output_file}")

if __name__ == "__main__":
    evenly_spaced_e(sample_idx=2, l=10, sigma=15)
