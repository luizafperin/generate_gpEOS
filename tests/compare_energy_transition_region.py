import numpy as np
import matplotlib.pyplot as plt

l = 15
sigma = 30
s = 2

# === Load data ===
file = f"/home/luizaperin/masters_research/gp/data_files/unconstrained/l{l}_s{sigma}/eos_hrg_s{s}_l{l}_s{sigma}.dat"  # replace with your file path
data = np.loadtxt(file, skiprows=1)

file =  f"/home/luizaperin/masters_research/gp/data_files/unconstrained/l{l}_s{sigma}/eos_pwr_s{s}_l{l}_s{sigma}.dat" # replace with your file path
data2 = np.loadtxt(file, skiprows=1)

file =f"/home/luizaperin/masters_research/gp/data_files/unconstrained/l{l}_s{sigma}/eos_w_s{s}_l{l}_s{sigma}.dat"  # replace with your file path
data3 = np.loadtxt(file, skiprows=1) 

# === Extract columns ===
e_h=data[:, 0]
T_h = data[:, 1]

e_p = data2[:, 0]
T_p= data2[:, 1]

e_w = data3[:,0]
T_w  = data3[:, 1]

plt.plot(T_h, e_h, '-', linewidth=1, color='black', label='HRG')
plt.plot(T_p, e_p , '-', linewidth=1, color='blue', label='PWR')
plt.plot(T_w, e_w , '-', linewidth=1, color='orange', label ='weighted')
#plt.axvspan(110, 120, color='gray', alpha=0.15, label='Transition region')
plt.xlabel('T [GeV]')
plt.ylabel('Energy Density [1/fm^4]')
plt.xlim(0.08, 0.16)
plt.ylim(0, 0.6)
plt.legend()
plt.title(f'Transition Between Extrapolations - Sample {s}')
plt.grid(True)

plt.tight_layout()
plt.savefig(f"/home/luizaperin/masters_research/hydro_res_analysis_run3/plots_unc/l{l}_s{sigma}/TransRegion_sample{s}")
plt.show()