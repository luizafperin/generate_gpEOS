# gp_sampler.py
"""
Gaussian Process Sampler for c_s^2 

This module provides classes and functions for generating samples of the speed of sound squared
using Gaussian Processes.

"""

import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import cholesky


class GPSampler:
    """
    A class for sampling c_s^2 using Gaussian Processes.

    Attributes:
        T_lattice (array): Temperature points from lattice data.
        cs2_lattice (array): c_s^2 values from lattice data.
        cs2_errors (array): Errors in lattice c_s^2.
        T_extracted (array): Temperature points from extracted data.
        cs2_extracted (array): c_s^2 values from extracted data.
        T_up (array): Temperature grid for high T extrapolation.
        T_total (array): Full temperature grid.
    """

    def __init__(self, lattice_cs2, extracted_cs2, T_up, T_total):
        """
        Initialize the GP sampler.

        Args:
            lattice_cs2 (DataFrame): Lattice data with columns [T, cs2, error].
            extracted_cs2 (DataFrame): Extracted data with columns [T, cs2].
            T_up (array): High T temperature grid (starting from where extracted data begins).
            T_total (array): Full T grid.
        """
        # Assume DataFrames with columns in order [T, cs2, error] for lattice, [T, cs2] for extracted
        self.T_lattice = lattice_cs2.iloc[:, 0].values
        self.cs2_lattice = lattice_cs2.iloc[:, 1].values
        self.cs2_errors = lattice_cs2.iloc[:, 2].values
        self.T_extracted = extracted_cs2.iloc[:, 0].values
        self.cs2_extracted = extracted_cs2.iloc[:, 1].values
        self.T_up = T_up
        self.T_total = T_total

        # Precompute mean data
        self.cs2_mean_lattice = self.cs2_lattice
        extracted_interp = interp1d(self.T_extracted, self.cs2_extracted, kind='linear', fill_value='extrapolate')
        self.cs2_mean_extracted = extracted_interp(self.T_up)
        self.mean_data = np.concatenate([self.cs2_mean_lattice, self.cs2_mean_extracted])

    @staticmethod
    def cs2_from_phi(phi):
        """Convert phi to c_s^2."""
        return 1 / (np.exp(phi) + 1)

    @staticmethod
    def phi_from_cs2(cs2):
        """Convert c_s^2 to phi."""
        return np.log(1 / cs2 - 1)

    @staticmethod
    def kernel_function(x1, x2, sigma, l):
        """Squared exponential kernel."""
        return sigma * np.exp(-(x1 - x2)**2 / (2 * l**2))

    def generate_sample(self, l, sigma):
        """
        Generate GP samples of c_s^2.

        Args:
            l (float): Correlation length.
            sigma (float): Variance.

        Returns:
            array: Sampled c_s^2 values on T_total.
        """
        # Transform to phi space
        mean_phi = np.array([self.phi_from_cs2(cs2) for cs2 in self.mean_data])

        # Build covariance matrix
        K = []
        for x1 in self.T_total:
            K_row = []
            for x2 in self.T_total:
                K_row.append(self.kernel_function(x1, x2, sigma, l))
            K.append(K_row)
        K = np.array(K)
        K[np.diag_indices_from(K)] += 0.001 

        # Cholesky decomposition
        L = cholesky(K)

        # Sample
        u = np.random.normal(0, 0.05, len(self.T_total))
        sample_phi = mean_phi + np.dot(L, u)

        # Transform back to c_s^2
        sample_cs2 = [self.cs2_from_phi(phi) for phi in sample_phi]

        return np.array(sample_cs2)
    
    def generate_sample_unconstrained(self, l, sigma, cte_mean):
        cs2_cte_mean = np.full((len(self.T_total)), cte_mean)
        K = []
        mean_phi = [self.phi_from_cs2(el) for el in cs2_cte_mean]
        for x1 in self.T_total:
            K_row = []
            for x2 in self.T_total:
                K_row.append(self.kernel_function(x1, x2, sigma, l))
            K.append(K_row)
        K = np.array(K)
        K[np.diag_indices_from(K)] += 0.001

        # Cholesky decomposition
        L = cholesky(K)

        # Sample
        u = np.random.normal(0, 0.05, len(self.T_total))
        sample_phi = mean_phi + np.dot(L, u)

        # Transform back to c_s^2
        sample_cs2 = [self.cs2_from_phi(phi) for phi in sample_phi]

        return np.array(sample_cs2)
    
    def generate_phi(self, l, sigma):
        """
        Generate phi samples 
        """
        # Transform to phi space
        mean_phi = np.array([self.phi_from_cs2(cs2) for cs2 in self.mean_data])

        # Build covariance matrix
        K = []
        for x1 in self.T_total:
            K_row = []
            for x2 in self.T_total:
                K_row.append(self.kernel_function(x1, x2, sigma, l))
            K.append(K_row)
        K = np.array(K)
        K[np.diag_indices_from(K)] += 0.001 

        # Cholesky decomposition
        L = cholesky(K)

        # Sample
        u = np.random.normal(0, 0.05, len(self.T_total))
        sample_phi = mean_phi + np.dot(L, u)

        return np.array(sample_phi)

