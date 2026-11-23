import argparse
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


def detect_category(filepath):
    """Detect whether an EOS file belongs to constrained or unconstrained data."""
    lower_path = filepath.lower()
    if "unconstrained" in lower_path:
        return "unconstrained"
    if "constrained" in lower_path:
        return "constrained"
    return "unknown"


def detect_tsw(filepath):
    """Detect switching temperature value from path segments like Tsw_136.

    Returns an int value when found (e.g. 136) or None otherwise.
    """
    parts = Path(filepath).parts
    for p in parts:
        m = re.match(r"Tsw_(\d+)$", p, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    m = re.search(r"Tsw_(\d+)", filepath, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def is_excluded_path(filepath, excluded_dirs=("Training_data",)):
    """Return True when filepath lives under an excluded directory."""
    path_parts = [part.lower() for part in Path(filepath).parts]
    return any(part in {name.lower() for name in excluded_dirs} for part in path_parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze EOS files with optional category, l, sigma, mode, and sample filters"
    )
    parser.add_argument(
        "--category",
        choices=["constrained", "unconstrained", "unknown", "all"],
        default="all",
        help="Choose constrained or unconstrained files (default: all)",
    )
    parser.add_argument("--l", type=int, default=None, help="Filter EOS files by l value")
    parser.add_argument("--sigma", type=int, default=None, help="Filter EOS files by sigma value")
    parser.add_argument("--mode", type=str, default=None, help="Filter EOS files by mode, e.g. hrg, pwr, w")
    parser.add_argument("--sample", type=int, default=None, help="Filter EOS files by sample index")
    parser.add_argument(
        "--input-root",
        default="gp/eosDataFiles",
        help="Root directory where EOS input files are located",
    )
    parser.add_argument(
        "--tsw",
        type=int,
        default=None,
        help="Filter by switching temperature value found in path (e.g. 136)",
    )
    parser.add_argument(
        "--base-output",
        default="/home/luizaperin/codes/eos_analysis_output",
        help="Base output directory for results",
    )
    parser.add_argument(
        "--include-training-data",
        action="store_true",
        help="Also include EOS files located under Training_data directories",
    )
    return parser.parse_args()


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
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.flatten()

    ylabels = [r"$\varepsilon$ [GeV/fm$^3$]", r"$p$ [GeV/fm$^3$]", r"$c_s^2$", r"$s$ [1/fm$^3$]"]

    sample_ids = []
    if compare_type == "samples":
        for index, label in enumerate(files_data):
            match = re.search(r"\d+", label)
            sample_ids.append(int(match.group()) if match else index)
        sample_norm = plt.Normalize(min(sample_ids) - 0.5, max(sample_ids) + 0.5)
        colormap = plt.get_cmap("viridis")
    else:
        colormap = plt.get_cmap("tab10")

    # Store data ranges for auto y-limit adjustment
    data_ranges = [[], [], [], []]  # for e, p, cs2, s

    for i, (label, (e, T, p, cs2, s)) in enumerate(files_data.items()):
        color = (
            colormap(sample_norm(sample_ids[i]))
            if compare_type == "samples"
            else colormap(i % colormap.N)
        )
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

    fig.suptitle(title, fontsize=14, y=0.99)
    if compare_type == "samples":
        sample_count = len(sample_ids)
        tick_step = max(1, int(np.ceil(sample_count / 8)))
        tick_positions = sample_ids[::tick_step]
        if sample_ids[-1] not in tick_positions:
            tick_positions.append(sample_ids[-1])
        scalar_mappable = plt.cm.ScalarMappable(norm=sample_norm, cmap=colormap)
        colorbar = fig.colorbar(
            scalar_mappable, ax=axs.tolist(), location="right", fraction=0.045, pad=0.04
        )
        colorbar.set_label("Sample index")
        colorbar.set_ticks(tick_positions)
        colorbar.set_ticklabels([str(sample_id) for sample_id in tick_positions])
        fig.subplots_adjust(right=0.88, top=0.92)
    else:
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=min(len(labels), 4),
                frameon=False,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.88 if handles else 0.95))
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

def main(args=None):
    if args is None:
        args = parse_args()

    # Find all eos_*.dat files under the selected input root
    input_root = os.path.abspath(args.input_root)
    print(f"Using input root: {input_root}")
    raw_eos_files = glob.glob(os.path.join(input_root, "eos_*.dat")) + glob.glob(os.path.join(input_root, "**/eos_*.dat"), recursive=True)
    eos_files = [
        path
        for path in raw_eos_files
        if args.include_training_data or not is_excluded_path(path)
    ]
    eos_files = sorted(set(eos_files))
    if not eos_files:
        print(f"No eos_*.dat files found under {input_root}.")
        return

    # Filter files and group by category, tsw, l, sigma
    groups = defaultdict(list)
    parsed_files = {}
    for filepath in eos_files:
        filename = os.path.basename(filepath)
        parsed = parse_filename(filename)
        if not parsed:
            continue

        mode, sample_idx, l, sigma = parsed
        category = detect_category(filepath)
        tsw_val = detect_tsw(filepath)

        if args.category != "all" and category != args.category:
            continue
        if args.tsw is not None:
            # If user requested a specific tsw but file doesn't have it, skip
            if tsw_val is None or tsw_val != args.tsw:
                continue
        if args.l is not None and l != args.l:
            continue
        if args.sigma is not None and sigma != args.sigma:
            continue
        if args.mode is not None and mode != args.mode:
            continue
        if args.sample is not None and sample_idx != args.sample:
            continue

        groups[(category, tsw_val, l, sigma)].append((filepath, mode, sample_idx))
        parsed_files[filepath] = (mode, sample_idx, l, sigma, category, tsw_val)

    if not parsed_files:
        print("No matching EOS files found for the selected filters.")
        return

    # Create base output directory
    output_base_dir = os.path.abspath(args.base_output)
    os.makedirs(output_base_dir, exist_ok=True)

    print("Selected filters:")
    print(f"  category = {args.category}")
    print(f"  tsw = {args.tsw}")
    print(f"  l = {args.l}")
    print(f"  sigma = {args.sigma}")
    print(f"  mode = {args.mode}")
    print(f"  sample = {args.sample}")
    print(f"  base output = {output_base_dir}")
    print(f"Found {len(parsed_files)} matching EOS file(s).\n")

    results = []
    for filepath, parsed in parsed_files.items():
        mode, sample_idx, l, sigma, category, tsw_val = parsed
        if tsw_val is not None:
            file_output_dir = os.path.join(output_base_dir, f"Tsw_{tsw_val}", category)
        else:
            file_output_dir = os.path.join(output_base_dir, category)
        os.makedirs(file_output_dir, exist_ok=True)
        result = analyze_eos_file(filepath, file_output_dir, (mode, sample_idx, l, sigma))
        results.append(result)

    # Now, for each category and group, do comparisons
    for (category, tsw_val, l, sigma), files in groups.items():
        if tsw_val is not None:
            group_dir = os.path.join(output_base_dir, f"Tsw_{tsw_val}", category, f"l{l}_s{sigma}")
        else:
            group_dir = os.path.join(output_base_dir, category, f"l{l}_s{sigma}")
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
                title_zoom = f"EOS Comparison - Modes (sample {sample_idx}, l={l}, sigma={sigma}) - T=0.10-0.16 GeV"
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
                plot_comparison(files_data, title, save_path, compare_type="samples")
                
                # Zoomed T range (100-160 GeV)
                title_zoom = f"EOS Comparison - Samples (mode {mode}, l={l}, sigma={sigma}) - T=0.10-0.16 GeV"
                save_path_zoom = os.path.join(samples_dir, f"samples_mode{mode}_zoom.png")
                plot_comparison(
                    files_data,
                    title_zoom,
                    save_path_zoom,
                    compare_type="samples",
                    xlim=(0.1, 0.16),
                )

    print(f"Analysis complete. Results saved in {output_base_dir}")

if __name__ == "__main__":
    main()