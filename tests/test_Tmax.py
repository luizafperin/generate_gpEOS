import re

def check_Tmax_increase(filename):
    T_values = []
    tau_values = []
    step_values = []

    with open(filename, "r") as f:
        lines = f.readlines()

    current_T = None

    for line in lines:
        # Extract T_max
        if "T_max =" in line:
            match = re.search(r"T_max = ([0-9.+-Ee]+)", line)
            if match:
                current_T = float(match.group(1))

        # Extract time step and tau
        if "Done time step" in line:
            step_match = re.search(r"time step (\d+)/\d+", line)
            tau_match = re.search(r"tau = ([0-9.+-Ee]+)", line)

            if step_match and tau_match and current_T is not None:
                step_values.append(int(step_match.group(1)))
                tau_values.append(float(tau_match.group(1)))
                T_values.append(current_T)

    # Check where T_max increases
    print("Checking for T_max increases:\n")
    found_increase = False

    for i in range(len(T_values) - 1):
        if T_values[i+1] > T_values[i]:
            found_increase = True
            print(f"Increase detected:")
            print(f"  Step {step_values[i]} → {step_values[i+1]}")
            print(f"  tau  {tau_values[i]} → {tau_values[i+1]} fm/c")
            print(f"  T_max {T_values[i]} → {T_values[i+1]} GeV\n")

    if not found_increase:
        print("No increases found. T_max is monotonically decreasing.")

# Example usage
check_Tmax_increase("/home/luizaperin/masters_research/hydro_res_analysis_run3/res_hrg/l10_s15/s23_l10_s15/s23_l10_s15.log")
