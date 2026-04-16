from matplotlib.pylab import multivariate_normal
import numpy as np
from scipy.interpolate import interp1d

""" ---------------------------------------------------------------------------------------------------------

                     Jackknife statistics, covariance, and correlation matrix calculations.

 -------------------------------------------------------------------------------------------------------------"""



def jackknife_statistics(data):
    """
    Obtain the covariance matrix from jackknife samples.
    data: array-like, shape (n_samples, n_features)
    Returns:
    cov_matrix: array, shape (n_features, n_features)
    """
    T = np.linspace(120,260,260-120+1)

    j_samples = data.iloc[:, 3:] #selects all lines and all columns after the 3rd
    jk_mean = j_samples.mean(axis=1) #mean over each row (axis=1), over the 48 samples for each T
    N = j_samples.shape[1]

    #transpose of j_sample (N_T, N_samples) to (N_samples, N_T), subtracts the mean from each column
    X = j_samples.T.sub(jk_mean, axis=1)  

    #calculate covariance matrix
    var_jk = ((N-1)/N) * ((j_samples.sub(jk_mean, axis=0)) ** 2).sum(axis=1)
    cov_matrix = (X.T @ X)*((N-1)/N)
    corr_matrix = cov_matrix / np.sqrt(np.outer(var_jk, var_jk))

    cov_matrix = cov_matrix.to_numpy()
    var_jk = var_jk.to_numpy()
    corr_matrix = corr_matrix.to_numpy()

    return {
        'cov_matrix': cov_matrix,
        'var': var_jk,
        'corr_matrix': corr_matrix,
        'mean': jk_mean
    }


""" ---------------------------------------------------------------------------------------------------------

                            Jackknife samples generation and extrapolations.

 -------------------------------------------------------------------------------------------------------------"""


def generate_jackknife_samples(n_samples, mean, cov):
    """
    Generate jackknife samples from input data (shape: n_samples x ...).
    Returns a list of jackknife samples, each with one element left out.
    """

    return multivariate_normal(mean=mean, cov=cov, size=n_samples)

def jkSamples_interp(n_samples, T, jk_samples):
    """
    Interpolate jackknife samples to the desired temperature range.
    Parameters:
    n_samples: int, number of jackknife samples
    T: array-like, temperature range for interpolation
    jk_samples: list of arrays, each array is a jackknife sample corresponding to the original T range
    Returns:
    list of interpolation functions for each jackknife sample
    """
    
    cs2_interp_jkLowT = []

    for i in range(n_samples):
    
        interp_func = interp1d(
            T,
            jk_samples[i],
            kind="linear",
            bounds_error=False, 
            fill_value=(jk_samples[i][0], jk_samples[i][-1])
        )
    
        cs2_interp_jkLowT.append(interp_func)

    return cs2_interp_jkLowT

""" ---------------------------------- Types of extrapolation functions-------------------------------------------------    """

def cs2_powerlaw(T, T0, interp):
    T = np.asarray(T)
    
    cs2_data = interp(T)
    cs2_T0 = interp(T0)

    A = T0**2 * (1.0/3.0 - cs2_T0)
    highT_part = 1.0/3.0 - A / T**2

    return np.where(T <= T0, cs2_data, highT_part)


def cs2_tanh(T, interp, T0=500.0, delta=70.0):
    T = np.asarray(T)
    w = 0.5 * (1 + np.tanh((T - T0) / delta))
    highT_part = (1 - w) * interp(T) + w * (1.0 / 3.0)
    return np.where(T <= T0, interp(T), highT_part)

def cs2_exp(T, interp, T0, X_inf=1.0/3.0, dT=0.5):
    """
    Exponential extrapolation of cs^2 that matches value and slope at T0.
    """
    T = np.asarray(T)

    # value at matching point
    y0 = interp(T0)

    # numerical slope at T0
    slope = (interp(T0 + dT) - interp(T0 - dT)) / (2*dT)

    delta = X_inf - y0

    # protect against division problems
    if abs(delta) < 1e-12 or abs(slope) < 1e-12:
        return np.full_like(T, X_inf)

    k = slope / delta

    def f(T):
        T = np.asarray(T)
        highTpart = X_inf - delta * np.exp(-k * (T - T0))
        return np.where(T <= T0, interp(T), highTpart)

    return f
