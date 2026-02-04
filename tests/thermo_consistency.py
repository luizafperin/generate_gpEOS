import numpy as np

def check_thermo_identity(
    filename,
    tol=1e-3,
    verbose=True
):
    """
    Checks e + p = s * T for an EOS / thermo table.

    Parameters
    ----------
    filename : str
        Path to the EOS file
    tol : float
        Absolute tolerance for |(e+p) - sT|
    verbose : bool
        Print discrepancies if True
    """

    # --- Load file
    with open(filename, "r") as f:
        lines = f.readlines()

    # --- First line: number of entries
    try:
        N = int(lines[0].strip())
    except ValueError:
        raise RuntimeError("First line must contain number of entries")

    # --- Load numeric data
    data = np.loadtxt(lines[1:1+N])

    if data.shape[1] != 5:
        raise RuntimeError("Expected 5 columns: e T p cs2 s")

    # --- Assign columns
    e   = data[:, 0]
    T   = data[:, 1]
    p   = data[:, 2]
    s   = data[:, 4]

    # --- Compute identity
    lhs = e + p
    rhs = s * T
    diff = np.abs(lhs - rhs)

    # --- Find violations
    bad = np.where(diff > tol)[0]

    if verbose:
        for i in bad:
            print(
                f"Discrepancy at i={i}: "
                f"T={T[i]:.6e} GeV | "
                f"e+p={lhs[i]:.6e}, s*T={rhs[i]:.6e}, "
                f"diff={diff[i]:.2e}"
            )

    return {
        "N": N,
        "n_bad": len(bad),
        "max_diff": diff.max(),
        "mean_diff": diff.mean(),
        "indices_bad": bad,
    }

result = check_thermo_identity(
    "./gp/data_files/constrained/l10_s15/eos_w_s23_l10_s15.dat",
    tol=1e-3
)

print("Summary:")
print("Total points:", result["N"])
print("Violations:", result["n_bad"])
print("Max diff:", result["max_diff"])
