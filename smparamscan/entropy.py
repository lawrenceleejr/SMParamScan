"""Entanglement entropy computation from Higgs branching ratios.

Based on arXiv:2511.17321 — linear entropy (Tsallis-2):
    EE = 1 - sum_i (P_i / N_c^i) * BR_i^2

where P_i is the spin factor and N_c^i is the color multiplicity.

Three physical regimes:
  - "partonic":    Full color factors (quarks N_c=3, gluons N_c=8).
                   This is the fundamental QFT calculation.
  - "hadronized":  Color confined -> N_c=1 for all channels.
                   Represents tracing out color after hadronization.
  - "inclusive":   Color confined + merge experimentally indistinguishable
                   light-hadron channels (gg + uu + dd + ss -> "light hadrons").
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


# Channels that produce indistinguishable light hadrons after hadronization
LIGHT_HADRON_CHANNELS = {"gg", "ss"}
# Note: uu, dd not in HDECAY output (massless), but ss is tiny too.
# We merge gg + ss (+ effectively uu, dd which have BR~0).
# cc and bb are distinguishable via displaced vertices / flavor tagging.


def compute_ee(branching_ratios: dict[str, float]) -> float:
    """Partonic EE — full color factors as in arXiv:2511.17321.

    EE = 1 - sum_i (P_i / N_c^i) * BR_i^2
    """
    spin = get_spin_factors()
    purity_sum = 0.0
    for ch in ALL_CHANNELS:
        br = branching_ratios.get(ch, 0.0)
        p_i = spin[ch]
        n_c = COLOR_FACTORS[ch]
        purity_sum += (p_i / n_c) * br**2

    return 1.0 - purity_sum


def compute_ee_hadronized(branching_ratios: dict[str, float]) -> float:
    """Post-hadronization EE — color confined, N_c=1 for all channels.

    After hadronization all final states are color singlets, so the color
    degree of freedom is no longer observable. This is equivalent to tracing
    out color from the density matrix before computing purity.

    EE_had = 1 - sum_i P_i * BR_i^2
    """
    spin = get_spin_factors()
    purity_sum = 0.0
    for ch in ALL_CHANNELS:
        br = branching_ratios.get(ch, 0.0)
        p_i = spin[ch]
        purity_sum += p_i * br**2

    return 1.0 - purity_sum


def compute_ee_inclusive(branching_ratios: dict[str, float]) -> float:
    """Inclusive hadronic EE — color confined + merge indistinguishable channels.

    After hadronization, gg and light-quark channels produce overlapping
    sets of light hadrons that are experimentally indistinguishable.
    We merge them into a single "light hadrons" channel.

    Heavy quarks (bb, cc) remain distinguishable via flavor tagging.
    Leptons (tautau, mumu) and bosons (WW, ZZ, gamgam, Zgam) are distinct.

    For merged channels with the same P_i = 1/2, the combined contribution
    to purity is P * BR_combined^2, where BR_combined = sum of individual BRs.
    This is larger than sum of individual P * BR_j^2 (by Cauchy-Schwarz),
    so EE_inclusive < EE_hadronized.
    """
    spin = get_spin_factors()
    purity_sum = 0.0

    # Accumulate BR for light-hadron channels
    br_light = 0.0

    for ch in ALL_CHANNELS:
        br = branching_ratios.get(ch, 0.0)
        if ch in LIGHT_HADRON_CHANNELS:
            br_light += br
        else:
            p_i = spin[ch]
            purity_sum += p_i * br**2  # N_c = 1 post-hadronization

    # Add merged light-hadron contribution (P = 1/2 for all sub-channels)
    purity_sum += 0.5 * br_light**2

    return 1.0 - purity_sum
