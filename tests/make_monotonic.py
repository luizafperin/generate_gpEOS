import numpy as np

def enforce_monotonic_and_save(
    filename,
    output_filename,
    column=0,
    delimiter=None,
    strict=False,
    tol=0.0,
    continuum_tol=1e-8
):
    # Load header separately
    with open(filename, "r") as f:
        header = f.readline()

    data = np.loadtxt(filename, delimiter=delimiter, skiprows=1)

    col = data[:, column]

    # Rows to keep: always keep the first row
    keep_mask = np.ones(len(data), dtype=bool)

    # Remove rows where first column values are ~constant or non-monotonic
    last_kept_value = col[0]
    removed_count = 0
    
    for i in range(1, len(data)):
        diff = col[i] - last_kept_value
        
        # Remove if difference is too small or negative (ensure strictly increasing)
        if diff <= continuum_tol:
            removed_count += 1
            keep_mask[i] = False
        else:
            # This row is kept, update last_kept_value
            last_kept_value = col[i]
    
    print(f"Removed {removed_count} rows due to insufficient increase in column {column}")

    cleaned_data = data[keep_mask]

    # Save cleaned file with updated header
    with open(output_filename, "w") as f:
        # Write the number of remaining rows as header
        f.write(f"{len(cleaned_data)}\n")
        # Use proper formatting to keep rows on single lines
        np.savetxt(f, cleaned_data, fmt='%.16e')

    print(f"Cleaned file written to: {output_filename}")
    print(f"Removed {np.sum(~keep_mask)} rows")

    return cleaned_data

cleaned = enforce_monotonic_and_save(
    "/home/luizaperin/masters_research/gp/data_files/unconstrained/l15_s30/eos_w_s2_l15_s30.dat",
    "/home/luizaperin/masters_research/gp/data_files/unconstrained/l15_s30/eos_w_s2_l15_s30_clean.dat",
    column=0,
    tol=1e-12
)



