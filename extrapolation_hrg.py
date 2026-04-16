import numpy as np
from scipy.interpolate import interp1d
from hrg import HRGCalculator

def extrapolate_hrg(P_down, P_up, e_down, e_up, s_down, s_up, T_total, cs2_interp, idx_Tsw):

    """
    HRG extrapolation up to switching temperature, then GP above.

    Parameters:
    T_total: array-like, total temperature range
    T_sw: float, switching temperature
    idx_Tsw: int, index of switching temperature in T_total
    cs2_interp: interpolation function for cs2 in upper range
    s_up_interp: interpolation function for entropy in upper range
    P_up_interp: interpolation function for pressure in upper range
    T_up: array-like, upper temperature range
    hrg_degrees_of_freedom: list, degrees of freedom for HRG [g_boson, g_fermion, g_boson_massless]

    Returns:
    dict with 'pressure', 'entropy', 'energy', 'cs2' arrays
    """

    # HRG up to T_sw
    T_hrg = T_total[:idx_Tsw]
    P_hrg = P_down[:idx_Tsw]
    e_hrg = e_down[:idx_Tsw]
    s_hrg = s_down[:idx_Tsw]

    # Calculate cs2 from HRG
    dP_dT_hrg = np.gradient(P_hrg, T_hrg)
    de_dT_hrg = np.gradient(e_hrg, T_hrg)

    with np.errstate(invalid='ignore', divide='ignore'):
        cs2_hrg = dP_dT_hrg / de_dT_hrg

    # Fill NaNs
    cs2_hrg[np.isnan(cs2_hrg)] = cs2_hrg[~np.isnan(cs2_hrg)][0] if np.any(~np.isnan(cs2_hrg)) else 1.0/3.0

    # GP above T_sw
    T_aboveTsw = T_total[idx_Tsw:]
    P_abvTsw = P_up(T_aboveTsw)
    s_abvTsw = s_up(T_aboveTsw)
    e_abvTsw = e_up(T_aboveTsw)
    cs2_abvTsw = cs2_interp(T_aboveTsw)

    # Concatenate
    P_total = np.concatenate([P_hrg, P_abvTsw])
    s_total = np.concatenate([s_hrg, s_abvTsw])
    e_total = np.concatenate([e_hrg, e_abvTsw])
    cs2_total = np.concatenate([cs2_hrg, cs2_abvTsw])

    return {
        'pressure': P_total,
        'entropy': s_total,
        'energy': e_total,
        'cs2': cs2_total
    }