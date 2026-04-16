import numpy as np
from scipy.interpolate import interp1d

def P_extrap_pwr(T, T_up, s_up_interp, P_up_interp):
    """
    Power law extrapolation for pressure downwards.

    Parameters:
    T: array-like, temperatures for extrapolation
    T_up: array-like, upper temperature range
    s_up_interp: interpolation function for entropy in upper range
    P_up_interp: interpolation function for pressure in upper range

    Returns:
    array-like, extrapolated pressure
    """
    T0 = T_up[0]
    s0 = s_up_interp(T0)
    p0 = P_up_interp(T0)

    n = (T0 * s0 / p0) - 1
    a_s = s0 / T0**n
    return (a_s * T**(n + 1)) / (n + 1)

def s_extrap_pwr(T, T_up, s_up_interp, P_up_interp):
    """
    Power law extrapolation for entropy downwards.

    Parameters:
    T: array-like, temperatures for extrapolation
    T_up: array-like, upper temperature range
    s_up_interp: interpolation function for entropy in upper range
    P_up_interp: interpolation function for pressure in upper range

    Returns:
    array-like, extrapolated entropy
    """
    T0 = T_up[0]
    s0 = s_up_interp(T0)
    p0 = P_up_interp(T0)

    n = (T0 * s0 / p0) - 1
    a_s = s0 / T0**n
    return a_s * T**n

def e_extrap_pwr(T, T_up, s_up_interp, P_up_interp):
    """
    Power law extrapolation for energy downwards.

    Parameters:
    T: array-like, temperatures for extrapolation
    T_up: array-like, upper temperature range
    s_up_interp: interpolation function for entropy in upper range
    P_up_interp: interpolation function for pressure in upper range

    Returns:
    array-like, extrapolated energy
    """
    s = s_extrap_pwr(T, T_up, s_up_interp, P_up_interp)
    P = P_extrap_pwr(T, T_up, s_up_interp, P_up_interp)
    return s * T - P

def cs2_extrap_pwr(T, T_up, s_up_interp, P_up_interp):
    """
    Power law extrapolation for cs2 downwards.

    Parameters:
    T: array-like, temperatures for extrapolation
    T_up: array-like, upper temperature range
    s_up_interp: interpolation function for entropy in upper range
    P_up_interp: interpolation function for pressure in upper range

    Returns:
    array-like, extrapolated cs2
    """
    T0 = T_up[0]
    s0 = s_up_interp(T0)
    p0 = P_up_interp(T0)

    n = (T0 * s0 / p0) - 1
    return np.full_like(T, 1.0 / n, dtype=float)

def extrapolate_powerlaw(T_down, T_up, T_total, cs2_interp, s_up_interp, P_up_interp):
    """
    Full power law extrapolation for thermodynamics.

    Parameters:
    T_down: array-like, down temperatures
    T_up: array-like, up temperatures
    T_total: array-like, total temperature range
    cs2_interp: interpolation function for cs2 in upper range
    s_up_interp: interpolation function for entropy in upper range
    P_up_interp: interpolation function for pressure in upper range
    T_transition_start: float, start of transition region

    Returns:
    dict with 'pressure', 'entropy', 'energy', 'cs2' arrays
    """


    # Extrapolate thermodynamics
    P_down = P_extrap_pwr(T_down, T_up, s_up_interp, P_up_interp)
    s_down = s_extrap_pwr(T_down, T_up, s_up_interp, P_up_interp)
    e_down = e_extrap_pwr(T_down, T_up, s_up_interp, P_up_interp)
    cs2_down = cs2_extrap_pwr(T_down, T_up, s_up_interp, P_up_interp)

    # Upper range
    P_up_vals = P_up_interp(T_up)
    s_up_vals = s_up_interp(T_up)
    e_up_vals = s_up_vals * T_up - P_up_vals
    cs2_up_vals = cs2_interp(T_up)

    # Concatenate
    P_total = np.concatenate([P_down, P_up_vals])
    s_total = np.concatenate([s_down, s_up_vals])
    e_total = np.concatenate([e_down, e_up_vals])
    cs2_total = np.concatenate([cs2_down, cs2_up_vals])


    # Interpolate cs2 over transition
    cs2_interp_total = interp1d(T_total, cs2_total, kind='linear')

    return {
        'pressure': P_total,
        'entropy': s_total,
        'energy': e_total,
        'cs2': cs2_interp_total(T_total)
    }