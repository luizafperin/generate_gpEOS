# hrg.py
"""
Hadron Resonance Gas (HRG) thermodynamics module.

This module provides functions for calculating pressure, energy, and entropy
using the HRG model with fermions and bosons.
"""

import numpy as np
from scipy.integrate import quad
from scipy.integrate import trapezoid
from scipy.special import kn


class HRGCalculator:
    """
    Calculator for HRG thermodynamics.

    Attributes:
        m_fermions (list): Masses of fermions in MeV.
        g_fermions (list): Degeneracies of fermions.
        m_bosons (list): Masses of bosons in MeV.
        g_bosons (list): Degeneracies of bosons.
    """

    def __init__(self, fermion_masses, fermion_degeneracies, boson_masses, boson_degeneracies):
        """
        Initialize HRG calculator.

        Args:
            fermion_masses (list): Masses of fermions.
            fermion_degeneracies (list): Degeneracies of fermions.
            boson_masses (list): Masses of bosons.
            boson_degeneracies (list): Degeneracies of bosons.
        """
        self.m_fermions = fermion_masses
        self.g_fermions = fermion_degeneracies
        self.m_bosons = boson_masses
        self.g_bosons = boson_degeneracies
        self.k_grid = np.linspace(0.0, 10000.0, 2000)

    @staticmethod
    def bose_einstein(k, T, mass):
        x = np.sqrt(k**2 + mass**2)/T
        x = np.float128(x)
        if x > 700:
            return 0.0
        return 1.0/(np.expm1(x))

    @staticmethod
    def fermi_dirac(k, T, mass):
        x = np.sqrt(k**2 + mass**2)/T
        x = np.float128(x)
        if x > 700:
            return 0.0
        return 1.0/(np.expm1(x) +1 + 1)

    def pressure_integrand(self, k, g, T, mass, eta):
        """Integrand for pressure."""
        if eta == 1:  # Fermion
            return (g / (6 * np.pi**2)) * k**4 * self.fermi_dirac(k, T, mass) / np.sqrt(k**2 + mass**2)
        elif eta == 0:  # Boson
            return (g / (6 * np.pi**2)) * k**4 * self.bose_einstein(k, T, mass) / np.sqrt(k**2 + mass**2)
        else:
            raise ValueError("eta must be 0 (boson) or 1 (fermion)")

    def energy_integrand(self, k, g, T, mass, eta):
        """Integrand for energy."""
        if eta == 1:  # Fermion
            return (g / (2 * np.pi**2)) * k**2 * np.sqrt(k**2 + mass**2) * self.fermi_dirac(k, T, mass)
        elif eta == 0:  # Boson
            return (g / (2 * np.pi**2)) * k**2 * np.sqrt(k**2 + mass**2) * self.bose_einstein(k, T, mass)
        else:
            raise ValueError("eta must be 0 (boson) or 1 (fermion)")

    def pressure_overT(self, T):
        """Calculate pressure for an array of temperatures using fixed-grid integration."""
        p_down = []
        for temp in T: 
            total = 0.0
            for m, g in zip(self.m_fermions, self.g_fermions):
                if temp <= 1.:
                    total += 0.0
                else:
                    result, _ = quad(self.pressure_integrand, 0, 10000, args=(g, temp, m, 1), limit=200)
                    total += result

            for m, g in zip(self.m_bosons, self.g_bosons):
                if temp <=1:
                    total += 0.0
                else:
                    result, _ = quad(self.pressure_integrand, 0, 10000, args=(g, temp, m, 0), limit=200)
                    total += result

            p_down.append(total)
        return np.array(p_down)  # MeV^4

    def energy_overT(self, T):
        e_down = []
        for temp in T: 
            total = 0.0     
            for m, g in zip(self.m_fermions, self.g_fermions):
                if temp <= 1:
                    total += 0.0
                else:
                    result, _ = quad(self.energy_integrand, 0, 10000, args=(g, temp+0.08, m, 1), limit=200)
                    total += result

            for m, g in zip(self.m_bosons, self.g_bosons):
                if temp <= 1:
                    total += 0.0
                else:
                    result, _ = quad(self.energy_integrand, 0, 10000, args=(g, temp+0.08, m, 0), limit=200)
                    total += result

            e_down.append(total)
        return np.array(e_down)  # MeV^4

    def entropy_overT(self, P, E, T):
        """Calculate entropy for an array of temperatures."""
        with np.errstate(divide='ignore', invalid='ignore'):
            S = np.where(T != 0, (E + P) / T, 0.0)
        return S

    def save_thermo_table(self, T, filename):
        """Compute P, E, S over T and save to a three-column text file."""
        T_arr = np.asarray(T, dtype=float)
        P = self.pressure_overT(T_arr)
        E = self.energy_overT(T_arr)
        S = self.entropy_overT(P, E, T_arr)

        data = np.column_stack((P, E, S))
        header = "P_hrg e_hrg s_hrg"
        np.savetxt(filename, data, header=header, fmt="%.8e")
        return filename

