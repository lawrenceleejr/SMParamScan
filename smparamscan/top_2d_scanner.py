"""2D scan of top quark and W boson decay EE over (m_W, m_t).

Varies both the W boson mass and the top quark mass simultaneously.
At each point computes:
  - W decay EE (depends on m_W; weak m_t dependence through tb threshold)
  - Top decay EE (depends on both m_t and m_W through the W propagator)

The W total width is computed self-consistently at each (m_W, m_t) point
and fed into the Breit-Wigner propagator for the top 3-body decay.
"""

import warnings

import numpy as np
from scipy import integrate

from .quark_decay import (
    _dalitz_s2_limits, _s2_integral,
    QUARK_MASSES, CKM2, W_LEPTONIC, W_HADRONIC_PAIRS, G_F,
)
from .w_decay import compute_w_ee, compute_w_total_width

SM_MW = 80.379
SM_MT = 172.5


def _breit_wigner_param(s, m_W, gamma_W):
    """W propagator squared with explicit m_W and Gamma_W."""
    return m_W**4 / ((s - m_W**2)**2 + m_W**2 * gamma_W**2)


def _top_partial_width(m_t, m_daughter, m_f1, m_f2,
                       V_prod_sq, V_decay_sq, Nc, m_W, gamma_W):
    """Partial width for t -> daughter + f1 + f2 with variable W parameters.

    Same Dalitz-plot integration as quark_decay.partial_width, but with
    m_W and gamma_W as explicit parameters rather than globals.
    """
    if m_t <= m_daughter + m_f1 + m_f2:
        return 0.0

    s1_min = (m_f1 + m_f2)**2
    s1_max = (m_t - m_daughter)**2

    def integrand(s1):
        s2_min, s2_max = _dalitz_s2_limits(s1, m_t, m_daughter, m_f1, m_f2)
        if s2_min is None:
            return 0.0
        I_s2 = _s2_integral(s2_min, s2_max, m_t, m_daughter, m_f1, m_f2)
        return _breit_wigner_param(s1, m_W, gamma_W) * I_s2

    # Tell the integrator about the BW peak so it can resolve narrow resonances
    s_peak = m_W**2
    points = [s_peak] if s1_min < s_peak < s1_max else []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        result, _ = integrate.quad(integrand, s1_min, s1_max,
                                   limit=200, points=points)

    prefactor = G_F**2 * V_prod_sq * V_decay_sq * Nc / (16 * np.pi**3 * m_t**3)
    return prefactor * result


def compute_top_decay_ee(m_t, m_W, gamma_W):
    """Compute top quark weak-decay EE at given (m_t, m_W, gamma_W).

    Returns:
        (ee, n_channels) tuple.
    """
    channels = []
    daughters = ["d", "s", "b"]

    for dq in daughters:
        m_dq = QUARK_MASSES[dq]
        V_prod_sq = CKM2.get(("t", dq), 0.0)
        if V_prod_sq == 0:
            continue

        # Leptonic W* modes
        for lep_name, m_l, m_nu in W_LEPTONIC:
            pw = _top_partial_width(m_t, m_dq, m_l, m_nu,
                                    V_prod_sq, 1.0, 1, m_W, gamma_W)
            if pw > 0:
                channels.append((f"{dq}+{lep_name}nu", pw))

        # Hadronic W* modes
        for hlabel, q1, q2 in W_HADRONIC_PAIRS:
            m_q1 = QUARK_MASSES[q1]
            m_q2 = QUARK_MASSES[q2]
            V_W_sq = CKM2.get((q1, q2), 0.0)
            pw = _top_partial_width(m_t, m_dq, m_q1, m_q2,
                                    V_prod_sq, V_W_sq, 3, m_W, gamma_W)
            if pw > 0:
                channels.append((f"{dq}+{hlabel}", pw))

    if not channels:
        return np.nan, 0

    total = sum(pw for _, pw in channels)
    if total <= 0:
        return np.nan, 0

    brs = {label: pw / total for label, pw in channels}
    ee = 1.0 - sum(br**2 for br in brs.values())
    return ee, len(channels)


def scan_2d(
    mw_min=5.0, mw_max=300.0, n_mw=50,
    mt_min=3.0, mt_max=500.0, n_mt=50,
):
    """2D scan over (m_W, m_t), computing W and top decay EE.

    Returns:
        mw_grid: 1D array of m_W values (GeV)
        mt_grid: 1D array of m_t values (GeV)
        w_ee: 2D array [n_mt, n_mw] of W decay EE
        top_ee: 2D array [n_mt, n_mw] of top decay EE
    """
    mw_grid = np.logspace(np.log10(mw_min), np.log10(mw_max), n_mw)
    mt_grid = np.logspace(np.log10(mt_min), np.log10(mt_max), n_mt)

    w_ee = np.full((n_mt, n_mw), np.nan)
    top_ee = np.full((n_mt, n_mw), np.nan)

    total = n_mt * n_mw
    count = 0

    for j, m_W in enumerate(mw_grid):
        for i, m_t in enumerate(mt_grid):
            # W decay EE (fast — 2-body computation)
            w_ee[i, j] = compute_w_ee(m_W, m_t)

            # W total width for the BW propagator in top decay
            gamma_W = compute_w_total_width(m_W, m_t)
            if gamma_W <= 0:
                gamma_W = 1e-6

            # Top decay EE (slower — Dalitz integration)
            top_ee[i, j], _ = compute_top_decay_ee(m_t, m_W, gamma_W)

            count += 1
            if count % 100 == 0 or count == 1:
                print(f"  [{count}/{total}] m_W={m_W:.1f} GeV, m_t={m_t:.1f} GeV, "
                      f"W_EE={w_ee[i, j]:.4f}, top_EE={top_ee[i, j]:.4f}")

    return mw_grid, mt_grid, w_ee, top_ee
