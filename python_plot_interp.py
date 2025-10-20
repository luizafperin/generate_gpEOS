
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



# Path to your output file
filename = "./interp_check.dat"   # adjust if needed

# Load data, skipping the header line that starts with '#'
data = np.loadtxt(filename, comments="#")

#index = data[:, 0]   # table index
e_val = data[:, 1]   # energy density
#T_tab = data[:, 2]   # tabulated T
T_int = data[:, 2]   # interpolated T
#diff  = data[:, 4]   # difference

p_py_int = interp1d(e_val, T_int, kind='linear', fill_value="extrapolate")
p_py_int_values = p_py_int(e_val)

# --- Plot T(e): tabulated vs interpolated
plt.figure(figsize=(7,5))
#plt.plot(e_val, T_tab, 'o', markersize=4, label="Tabulated T(e)")
plt.plot(e_val, T_int, 'o', markersize=4, label="Interpolation defined by hand, random energy points")
plt.plot(e_val, p_py_int_values, '--', linewidth=1.2, label="SciPy Interpolated")
plt.xlabel(r"Entropy density $e$ [1/fm$^4$]")
plt.ylabel(r"Temperature $T$ [1/fm]")
plt.title("Interpolation check")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("T_vs_s.png")
plt.show()

# ---
