import numpy as np

# === parameters ===
l = 10          # set your value
sigma = 15      # set your value
sample_idx = 20  # set your value

T_target_GeV = 0.120  # 120 MeV
tol = 1e-4            # tolerance in GeV

base_path = f"./gp/data_files/constrained/l{l}_s{sigma}"

files = {
    "pwr": f"{base_path}/eos_pwr_s{sample_idx}_l{l}_s{sigma}.dat",
    "w":   f"{base_path}/eos_w_s{sample_idx}_l{l}_s{sigma}.dat",
    "hrg": f"{base_path}/eos_hrg_s{sample_idx}_l{l}_s{sigma}.dat",
}

def get_energy_at_T(filename, T_target, tol):
    # skip first line (number of entries)
    data = np.loadtxt(filename, skiprows=1)

    energy = data[:, 0]  # GeV/fm^3
    T = data[:, 1]       # GeV

    idx = np.argmin(np.abs(T - T_target))

    if abs(T[idx] - T_target) > tol:
        raise ValueError(
            f"No temperature close enough to {T_target} GeV in {filename}"
        )

    return energy[idx], T[idx]

# === read energies ===
energies = {}

for key, fname in files.items():
    e, T_found = get_energy_at_T(fname, T_target_GeV, tol)
    energies[key] = e
    print(f"{key}: T = {T_found:.6f} GeV, e = {e:.6e} GeV/fm^3")

# === differences ===
print("\nEnergy differences at T = 120 MeV:")
print(f"pwr - w   = {energies['pwr'] - energies['w']:.6e} GeV/fm^3")
print(f"pwr - hrg = {energies['pwr'] - energies['hrg']:.6e} GeV/fm^3")
print(f"w   - hrg = {energies['w']   - energies['hrg']:.6e} GeV/fm^3")
