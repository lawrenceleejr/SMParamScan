"""Entanglement entropy computation from Higgs branching ratios.

Based on arXiv:2511.17321 — linear entropy (Tsallis-2):
    EE = 1 - sum_i (P_i / N_c^i) * BR_i^2

where P_i is the spin factor and N_c^i is the color multiplicity.
"""

import numpy as np
from scipy import integrate

from .config import ALL_CHANNELS, SPIN_FACTORS, COLOR_FACTORS, M_HIGGS, M_W, M_Z


def _compute_spin_factor_3body(m_h: float, m_v: float) -> float:
    """Compute the spin factor P_VV for off-shell h -> VV* -> V ff'.

    Uses the three-body formulas from arXiv:2511.17321 eqs. 19-21:
        P_VV = (2 F_T^2 + F_L^2) / (2 F_T + F_L)^2

        F_T(eps) = int_0^{(eps-1)^2} dy  y/(y-1)^2  sqrt(Y - 4y)
        F_L(eps) = int_0^{(eps-1)^2} dy  1/(4(y-1)^2)  Y sqrt(Y - 4y)

    where Y = (eps^2 - 1 - y)^2  and  eps = m_h / m_V.
    """
    eps = m_h / m_v
    y_max = (eps - 1.0) ** 2

    def Y_func(y):
        return (eps**2 - 1.0 - y) ** 2

    def integrand_T(y):
        Yval = Y_func(y)
        arg = Yval - 4.0 * y
        if arg <= 0:
            return 0.0
        return (y / (y - 1.0) ** 2) * np.sqrt(arg)

    def integrand_L(y):
        Yval = Y_func(y)
        arg = Yval - 4.0 * y
        if arg <= 0:
            return 0.0
        return (1.0 / (4.0 * (y - 1.0) ** 2)) * Yval * np.sqrt(arg)

    F_T, _ = integrate.quad(integrand_T, 0, y_max, limit=200)
    F_L, _ = integrate.quad(integrand_L, 0, y_max, limit=200)

    denominator = (2.0 * F_T + F_L) ** 2
    if denominator == 0:
        return 1.0 / 3.0  # fallback to equal polarization
    return (2.0 * F_T**2 + F_L**2) / denominator


# Pre-compute spin factors for WW and ZZ at the SM Higgs mass
P_WW = _compute_spin_factor_3body(M_HIGGS, M_W)
P_ZZ = _compute_spin_factor_3body(M_HIGGS, M_Z)


def get_spin_factors() -> dict[str, float]:
    """Return complete spin factor dict with WW/ZZ values filled in."""
    factors = dict(SPIN_FACTORS)
    factors["WW"] = P_WW
    factors["ZZ"] = P_ZZ
    return factors


def compute_ee(branching_ratios: dict[str, float]) -> float:
    """Compute the entanglement entropy from branching ratios.

    EE = 1 - sum_i (P_i / N_c^i) * BR_i^2

    Args:
        branching_ratios: dict mapping channel name -> BR value.

    Returns:
        The linear entanglement entropy (dimensionless, in [0, 1]).
    """
    spin = get_spin_factors()
    purity_sum = 0.0
    for ch in ALL_CHANNELS:
        br = branching_ratios.get(ch, 0.0)
        p_i = spin[ch]
        n_c = COLOR_FACTORS[ch]
        purity_sum += (p_i / n_c) * br**2

    return 1.0 - purity_sum
