import numpy as np
from scipy.interpolate import interp1d

""" ---------------------------------------------------------------------------------------------------------

                Weighted interpolation function to smoothly connect T-low and T-high regions 

 -------------------------------------------------------------------------------------------------------------"""

def weighted_interp(x, val_down, val_up):
    """
    Linear interpolation between two values.

    Parameters:
    x: float, weight (0 to 1)
    val_down: float, last value at the downwards region
    val_up: float, first value at the upper region

    Returns:
    float, interpolated value
    """
    return (1 - x) * val_down + x * val_up

def weighted_linearInterp(vec_down, vec_up, n_transition, T_total):
    """
    Weighted linear extrapolation for merging down and up vectors.

    Parameters:
    vec_down: array-like, values in down range
    vec_up: array-like, values in up range
    n_transition: int, number of points in transition region

    Returns:
    array-like, merged vector
    """
    vec_down = np.asarray(vec_down, dtype=float)
    vec_up = np.asarray(vec_up, dtype=float)

    up_start = vec_up[0]

    # Find last index in vec_down that is strictly below vec_up[0]
    valid_indices = np.where(vec_down < up_start)[0]

    if len(valid_indices) == 0:
        cut_idx = 0
    elif vec_down[-1] > up_start:
        cut_idx = valid_indices[-15]  # Use last 15 points or adjust as needed
    else:
        cut_idx = valid_indices[-1] + 1

    vec_down_cut = vec_down[:cut_idx]

    # Define total size
    N_total = len(vec_down) + n_transition + len(vec_up)

    # Transition endpoints
    start_val = vec_down_cut[-1] if len(vec_down_cut) > 0 else up_start
    end_val = up_start

    # Adjust transition size
    n_transition_original = n_transition

    n_transition = N_total - len(vec_down_cut) - len(vec_up)

    if n_transition != n_transition_original:
        print(f"n_transition adjusted to {n_transition}")

    # Weights
    w_range = np.linspace(0, 1, n_transition, endpoint=False)
    diff = N_total - (len(vec_down_cut) + n_transition + len(vec_up))
    
    if diff != 0:
        print('the difference should be zero, but is = ' + str(diff))

    # Interpolate transition
    transition = np.array([weighted_interp(w, start_val, end_val) for w in w_range])

    # Concatenate
    vec_total = np.concatenate([vec_down_cut, transition, vec_up])

    # Interpolate to ensure correct length
    interpolation = interp1d(T_total, vec_total, kind='linear')

    return {'result': interpolation(T_total),
            'adjusted_transition': n_transition}


""" ---------------------------------------------------------------------------------------------------------

                                                     Function to calculate cs2 

 -------------------------------------------------------------------------------------------------------------"""


def cs2_interp_w(cs2_interp, P, e, T_total, T_up, endpointIdx):
    dP_dT_to120 = np.gradient(P[:endpointIdx], T_total[:endpointIdx])
    de_dT_to120 = np.gradient(e[:endpointIdx], T_total[:endpointIdx])

    if np.isnan(dP_dT_to120).any():
        print("Warning: NaN values found in dP/dT. For P index = " + str(np.where(np.isnan(dP_dT_to120))))
    if np.isnan(de_dT_to120).any():
        print("Warning: NaN values found in de/dT. For e index = " + str(np.where(np.isnan(de_dT_to120))))

    with np.errstate(invalid='ignore', divide='ignore'): 
        cs2_to120 = dP_dT_to120 / de_dT_to120

    cs2_to120[np.isnan(cs2_to120)] = cs2_to120[~np.isnan(cs2_to120)][0]

    cs2_total_w = np.concatenate([cs2_to120, cs2_interp(T_up)])

    cs2_interp_w = interp1d(T_total, cs2_total_w, kind='linear', fill_value="extrapolate")

    return cs2_interp_w(T_total)

""" ---------------------------------------------------------------------------------------------------------

                     Full weighted extrapolation for thermodynamics, merging HRG and sample values

 -------------------------------------------------------------------------------------------------------------"""


def extrapolate_weighted(P_down, P_up, e_down, e_up, T_up, T_total, cs2_interp, n_transition, endpointIdx):
    """
    Weighted extrapolation for thermodynamics.

    Parameters:
    P_down: array-like, pressure in down range
    P_up: array-like, pressure in up range
    e_down: array-like, energy in down range
    e_up: array-like, energy in up range
    T_total: array-like, total temperature range
    cs2_interp: function, interpolated cs2 values
    n_transition: int, number of points in transition region

    Returns:
    dict with 'pressure', 'entropy', 'energy', 'cs2' arrays
    """
    P = weighted_linearInterp(P_down, P_up, n_transition, T_total)
    P_total = P['result']
    n_transition_P = P['adjusted_transition']
    e = weighted_linearInterp(e_down, e_up, n_transition, T_total)
    e_total = e['result']
    n_transition_e = e['adjusted_transition']

    # Entropy from e + P = T*s
    s_total = np.where(T_total != 0, (P_total + e_total) / T_total, 0.0)

    # cs2 from dP/de
    cs2_total = cs2_interp_w(cs2_interp, P_total, e_total, T_total, T_up, endpointIdx)

    return {
        'pressure': P_total,
        'entropy': s_total,
        'energy': e_total,
        'cs2': cs2_total,
        'adjusted_transition_P': n_transition_P,
        'adjusted_transition_e': n_transition_e
    }