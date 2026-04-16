import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import re
from collections import defaultdict

# Set plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["font.size"] = 10

def parse_filename(filename):
    """Parse eos filename to extract mode, sample_idx, l, sigma."""
    match = re.match(r"eos_(\w+)_s(\d+)_l400_s(\d+)\.dat", filename)
    if match:
        mode, sample_idx, sigma = match.groups()
        return mode, int(sample_idx), 400, int(sigma)
    return None

def load_eos_file(filename):
    """Load an EOS file and return arrays for e, T, p, cs2, and s."""
    with open(filename, "r") as f:
        lines = f.readlines()
    # First line is number of entries
    N = int(lines[0].strip())
    data = np.loadtxt(lines[1:1+N])
    e, T, p, cs2, s = data.T
    return e, T, p, cs2, s

def check_thermo_identity(filename, tol=1e-3, verbose=False):
    """
    Checks e + p = s * T for an EOS file.

    Returns dict with summary.
    """
    e, T, p, cs2, s = load_eos_file(filename)
    lhs = e + p
    rhs = s * T
    diff = np.abs(lhs - rhs)
    bad = np.where(diff > tol)[0]
    if verbose:
        for i in bad:
            print(f"Discrepancy at i={i}: T={T[i]:.6e}, e+p={lhs[i]:.6e}, s*T={rhs[i]:.6e}, diff={diff[i]:.2e}")
    return {
        "N": len(e),
        "n_bad": len(bad),
        "max_diff": diff.max(),
        "mean_diff": diff.mean(),
        "indices_bad": bad,
    }

def enforce_monotonic(data, column=0, tol=1e-8):
    """Enforce monotonic increase in a column, return cleaned data."""
    col = data[:, column]
    keep_mask = np.ones(len(data), dtype=bool)
    last_kept_value = col[0]
    for i in range(1, len(data)):
        diff = col[i] - last_kept_value
        if diff <= tol:
            keep_mask[i] = False
        else:
            last_kept_value = col[i]
    return data[keep_mask]

def plot_eos_quantities(e, T, p, cs2, s, title, save_path, xlim=None):
    """Plot the EOS quantities."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    axs[0].plot(T, e, 'b-')
    axs[0].set_ylabel(r"$\varepsilon$ [GeV/fm$^3$]")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(T, p, 'r-')
    axs[1].set_ylabel(r"$p$ [GeV/fm$^3$]")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(T, cs2, 'g-')
    axs[2].set_ylabel(r"$c_s^2$")
    axs[2].set_xlabel(r"$T$ [GeV]")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(T, s, 'm-')
    axs[3].set_ylabel(r"$s$ [1/fm$^3$]")
    axs[3].set_xlabel(r"$T$ [GeV]")
    axs[3].grid(True, alpha=0.3)

    if xlim:
        mask = (T >= xlim[0]) & (T <= xlim[1])
        e_plot = e[mask]
        p_plot = p[mask]
        cs2_plot = cs2[mask]
        s_plot = s[mask]
        for ax in axs:
            ax.set_xlim(xlim)
    else:
        e_plot = e
        p_plot = p
        cs2_plot = cs2
        s_plot = s

    data_ranges = [e_plot, p_plot, cs2_plot, s_plot]
    for i, ax in enumerate(axs):
        if len(data_ranges[i]) > 0:
            data_min = np.min(data_ranges[i])
            data_max = np.max(data_ranges[i])
            padding = (data_max - data_min) * 0.1
            if padding == 0:
                padding = abs(data_min) * 0.1 if data_min != 0 else 1
            ax.set_ylim(data_min - padding, data_max + padding)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_comparison(files_data, title, save_path, compare_type="modes", xlim=None):
    """Plot comparison of multiple EOS files.
    
    Parameters
    ----------
    files_data : dict
        Dictionary with labels as keys and (e, T, p, cs2, s) tuples as values
    title : str
        Plot title
    save_path : str
        Path to save figure
    compare_type : str
        Type of comparison ("modes" or "samples")
    xlim : tuple or None
        X-axis limits (T_min, T_max)
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    quantities = ['e', 'p', 'cs2', 's']
    ylabels = [r"$\varepsilon$ [GeV/fm$^3$]", r"$p$ [GeV/fm$^3$]", r"$c_s^2$", r"$s$ [1/fm$^3$]"]
    
    # Store data ranges for auto y-limit adjustment
    data_ranges = [[], [], [], []]  # for e, p, cs2, s
    
    for i, (label, (e, T, p, cs2, s)) in enumerate(files_data.items()):
        color = colors[i % len(colors)]
        axs[0].plot(T, e, label=label, color=color, lw=1)
        axs[1].plot(T, p, label=label, color=color, lw=1)
        axs[2].plot(T, cs2, label=label, color=color, lw=1)
        axs[3].plot(T, s, label=label, color=color, lw=1)
        
        # Collect data for y-limit calculation
        if xlim:
            mask = (T >= xlim[0]) & (T <= xlim[1])
            data_ranges[0].extend(e[mask])
            data_ranges[1].extend(p[mask])
            data_ranges[2].extend(cs2[mask])
            data_ranges[3].extend(s[mask])
        else:
            data_ranges[0].extend(e)
            data_ranges[1].extend(p)
            data_ranges[2].extend(cs2)
            data_ranges[3].extend(s)

    for ax, ylabel in zip(axs, ylabels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if xlim:
            ax.set_xlim(xlim)

    # Auto-adjust y-limits with some padding
    for i, ax in enumerate(axs):
        if data_ranges[i]:
            data_min = np.min(data_ranges[i])
            data_max = np.max(data_ranges[i])
            # Add 10% padding
            padding = (data_max - data_min) * 0.1
            if padding == 0:
                padding = abs(data_min) * 0.1 if data_min != 0 else 1
            ax.set_ylim(data_min - padding, data_max + padding)

    axs[2].set_xlabel(r"$T$ [GeV]")
    axs[3].set_xlabel(r"$T$ [GeV]")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def analyze_eos_file(filepath, output_dir, parsed):
    """Analyze a single EOS file."""
    filename = os.path.basename(filepath)
    mode, sample_idx, l, sigma = parsed
    print(f"Analyzing {filename}")

    # Load data
    e, T, p, cs2, s = load_eos_file(filepath)

    # Check thermo consistency
    thermo_result = check_thermo_identity(filepath, tol=1e-3, verbose=False)

    # Make monotonic if needed (on energy)
    data = np.column_stack((e, T, p, cs2, s))
    original_len = len(data)
    cleaned_data = enforce_monotonic(data, column=0)
    removed = original_len - len(cleaned_data)

    # Save cleaned file if changed
    if removed > 0:
        monotonic_dir = os.path.join(output_dir, "monotonic_files")
        os.makedirs(monotonic_dir, exist_ok=True)
        cleaned_filepath = os.path.join(monotonic_dir, filename)
        with open(cleaned_filepath, "w") as f:
            f.write(f"{len(cleaned_data)}\n")
            np.savetxt(f, cleaned_data, fmt='%.16e')
        print(f"  Cleaned file saved: {cleaned_filepath}, removed {removed} points")

    # Create sample dir
    sample_dir = os.path.join(output_dir, f"l{l}_s{sigma}", f"res_{mode}", f"s{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    # Plot individual (whole T)
    plot_path_full = os.path.join(sample_dir, f"{filename.replace('.dat', '_full.png')}")
    plot_eos_quantities(e, T, p, cs2, s, f"EOS: {filename}", plot_path_full)

    # Plot individual (zoomed T=100-160)
    plot_path_zoom = os.path.join(sample_dir, f"{filename.replace('.dat', '_zoom.png')}")
    plot_eos_quantities(e, T, p, cs2, s, f"EOS: {filename} (T=100-160 GeV)", plot_path_zoom, xlim=(0.1, 0.16))

    # Log per sample
    log_path = os.path.join(sample_dir, f"{filename.replace('.dat', '.log')}")
    with open(log_path, "w") as log:
        log.write(f"Analysis for {filename}\n")
        log.write("=" * 40 + "\n")
        log.write(f"Mode: {mode}, Sample: {sample_idx}, l: {l}, sigma: {sigma}\n")
        log.write(f"Total points: {thermo_result['N']}\n")
        log.write(f"Thermo violations: {thermo_result['n_bad']}\n")
        log.write(f"Max diff: {thermo_result['max_diff']:.2e}\n")
        log.write(f"Mean diff: {thermo_result['mean_diff']:.2e}\n")
        log.write(f"Points removed for monotonicity: {removed}\n")

    return {
        "filename": filename,
        "thermo": thermo_result,
        "removed_points": removed,
        "data": (e, T, p, cs2, s),
    }

def main():
    # Find all eos_*.dat files
    eos_files = glob.glob("eos_*.dat") + glob.glob("**/eos_*.dat", recursive=True)
    if not eos_files:
        print("No eos_*.dat files found.")
        return

    # Group files by (l, sigma)
    groups = defaultdict(list)
    parsed_files = {}
    for filepath in eos_files:
        filename = os.path.basename(filepath)
        parsed = parse_filename(filename)
        if parsed:
            mode, sample_idx, l, sigma = parsed
            groups[(l, sigma)].append((filepath, mode, sample_idx))
            parsed_files[filepath] = parsed

    # Create output directory
    output_dir = "/home/luizaperin/masters_research/gp/eos_analysis_output"
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for filepath in eos_files:
        parsed = parsed_files.get(filepath)
        if parsed:
            result = analyze_eos_file(filepath, output_dir, parsed)
            results.append(result)

    # Now, for each group, do comparisons
    for (l, sigma), files in groups.items():
        group_dir = os.path.join(output_dir, f"l{l}_s{sigma}")
        os.makedirs(group_dir, exist_ok=True)

        # Group by sample_idx
        samples = defaultdict(dict)
        for filepath, mode, sample_idx in files:
            samples[sample_idx][mode] = filepath

        # For each sample, compare modes
        for sample_idx, modes_dict in samples.items():
            if len(modes_dict) > 1:
                files_data = {}
                for mode, fp in modes_dict.items():
                    e, T, p, cs2, s = load_eos_file(fp)
                    files_data[mode] = (e, T, p, cs2, s)
                
                # Full T range
                title = f"EOS Comparison - Modes (sample {sample_idx}, l={l}, sigma={sigma})"
                modes_dir = os.path.join(group_dir, "modes_comparison")
                os.makedirs(modes_dir, exist_ok=True)
                save_path = os.path.join(modes_dir, f"modes_sample{sample_idx}_full.png")
                plot_comparison(files_data, title, save_path)
                
                # Zoomed T range (100-160 GeV)
                title_zoom = f"EOS Comparison - Modes (sample {sample_idx}, l={l}, sigma={sigma}) - T=100-160 GeV"
                save_path_zoom = os.path.join(modes_dir, f"modes_sample{sample_idx}_zoom.png")
                plot_comparison(files_data, title_zoom, save_path_zoom, xlim=(0.1, 0.16))

        # For each mode, compare samples
        modes = defaultdict(dict)
        for filepath, mode, sample_idx in files:
            modes[mode][sample_idx] = filepath

        for mode, samples_dict in modes.items():
            if len(samples_dict) > 1:
                files_data = {}
                for sample_idx, fp in samples_dict.items():
                    e, T, p, cs2, s = load_eos_file(fp)
                    files_data[f"s{sample_idx}"] = (e, T, p, cs2, s)
                
                # Full T range
                title = f"EOS Comparison - Samples (mode {mode}, l={l}, sigma={sigma})"
                samples_dir = os.path.join(group_dir, "samples_comparison")
                os.makedirs(samples_dir, exist_ok=True)
                save_path = os.path.join(samples_dir, f"samples_mode{mode}_full.png")
                plot_comparison(files_data, title, save_path)
                
                # Zoomed T range (100-160 GeV)
                title_zoom = f"EOS Comparison - Samples (mode {mode}, l={l}, sigma={sigma}) - T=100-160 GeV"
                save_path_zoom = os.path.join(samples_dir, f"samples_mode{mode}_zoom.png")
                plot_comparison(files_data, title_zoom, save_path_zoom, xlim=(0.1, 0.16))

    print(f"Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()