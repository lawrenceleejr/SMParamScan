"""Systematic study of quantities that might show features at y_t = 1.

The top quark is unique: y_t = sqrt(2) m_t / v ≈ 0.991. This module scans
many different quantities as functions of y_t, looking for extrema,
inflection points, or other special features near y = 1.
"""

import warnings

import numpy as np
from scipy import integrate

from .quark_decay import (
    _dalitz_s2_limits, _s2_integral,
    QUARK_MASSES, CKM2, W_LEPTONIC, W_HADRONIC_PAIRS, G_F,
)
from .w_decay import compute_w_total_width, compute_w_ee

V_HIGGS = 246.22  # GeV
SM_MW = 80.379
SM_MT = 172.5
M_B = QUARK_MASSES["b"]


def _breit_wigner(s, m_W, gamma_W):
    return m_W**4 / ((s - m_W**2)**2 + m_W**2 * gamma_W**2)


def _partial_width(m_t, m_daughter, m_f1, m_f2,
                   V_prod_sq, V_decay_sq, Nc, m_W, gamma_W):
    """Top partial width with explicit W parameters."""
    if m_t <= m_daughter + m_f1 + m_f2:
        return 0.0
    s1_min = (m_f1 + m_f2)**2
    s1_max = (m_t - m_daughter)**2

    def integrand(s1):
        s2_min, s2_max = _dalitz_s2_limits(s1, m_t, m_daughter, m_f1, m_f2)
        if s2_min is None:
            return 0.0
        I_s2 = _s2_integral(s2_min, s2_max, m_t, m_daughter, m_f1, m_f2)
        return _breit_wigner(s1, m_W, gamma_W) * I_s2

    s_peak = m_W**2
    points = [s_peak] if s1_min < s_peak < s1_max else []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _ = integrate.quad(integrand, s1_min, s1_max,
                                   limit=200, points=points)
    prefactor = G_F**2 * V_prod_sq * V_decay_sq * Nc / (16 * np.pi**3 * m_t**3)
    return prefactor * result


def compute_top_channels(m_t, m_W=SM_MW, gamma_W=None):
    """Compute all top decay channels and return detailed info.

    Returns dict with:
      channels: list of (label, partial_width)
      brs: dict of label -> BR
      total_width: float
      n_channels: int
    """
    if gamma_W is None:
        gamma_W = compute_w_total_width(m_W, m_t)
        if gamma_W <= 0:
            gamma_W = 1e-6

    channels = []
    for dq in ["d", "s", "b"]:
        m_dq = QUARK_MASSES[dq]
        V_prod_sq = CKM2.get(("t", dq), 0.0)
        if V_prod_sq == 0:
            continue
        for lep_name, m_l, m_nu in W_LEPTONIC:
            pw = _partial_width(m_t, m_dq, m_l, m_nu,
                                V_prod_sq, 1.0, 1, m_W, gamma_W)
            if pw > 0:
                channels.append((f"{dq}+{lep_name}nu", pw))
        for hlabel, q1, q2 in W_HADRONIC_PAIRS:
            m_q1, m_q2 = QUARK_MASSES[q1], QUARK_MASSES[q2]
            V_W_sq = CKM2.get((q1, q2), 0.0)
            pw = _partial_width(m_t, m_dq, m_q1, m_q2,
                                V_prod_sq, V_W_sq, 3, m_W, gamma_W)
            if pw > 0:
                channels.append((f"{dq}+{hlabel}", pw))

    if not channels:
        return {"channels": [], "brs": {}, "total_width": 0.0, "n_channels": 0}

    total = sum(pw for _, pw in channels)
    brs = {label: pw / total for label, pw in channels}
    return {
        "channels": channels,
        "brs": brs,
        "total_width": total,
        "n_channels": len(channels),
    }


def compute_all_quantities(y_t, m_W=SM_MW):
    """Compute all interesting quantities at a given y_t value.

    Returns dict of quantity_name -> value.
    """
    m_t = y_t * V_HIGGS / np.sqrt(2)
    gamma_W = compute_w_total_width(m_W, m_t)
    if gamma_W <= 0:
        gamma_W = 1e-6

    info = compute_top_channels(m_t, m_W, gamma_W)
    brs = info["brs"]
    total_width = info["total_width"]
    n_ch = info["n_channels"]

    result = {
        "y_t": y_t,
        "m_t": m_t,
        "kappa_t": m_t / SM_MT,
    }

    if not brs or total_width <= 0:
        for key in ["ee_linear", "ee_shannon", "ee_renyi2", "n_eff",
                     "total_width", "width_over_mass", "width_over_GammaW",
                     "n_channels", "max_br", "br_ratio_12", "kl_from_uniform",
                     "ee_w", "joint_ee", "mt_over_mw", "onshell_margin",
                     "phase_space_factor"]:
            result[key] = np.nan
        return result

    br_vals = np.array(list(brs.values()))

    # === Linear entropy (Tsallis-2) ===
    ee_linear = 1.0 - np.sum(br_vals**2)

    # === Shannon entropy ===
    br_safe = br_vals[br_vals > 0]
    ee_shannon = -np.sum(br_safe * np.log(br_safe))

    # === Rényi-2 entropy ===
    ee_renyi2 = -np.log(np.sum(br_vals**2))

    # === Effective number of channels ===
    n_eff = 1.0 / np.sum(br_vals**2)  # = 1/(1 - EE_linear)

    # === Width quantities ===
    width_over_mass = total_width / m_t
    width_over_GammaW = total_width / gamma_W

    # === BR structure ===
    sorted_brs = np.sort(br_vals)[::-1]
    max_br = sorted_brs[0]
    br_ratio_12 = sorted_brs[0] / sorted_brs[1] if len(sorted_brs) > 1 else np.inf

    # === KL divergence from uniform ===
    uniform = 1.0 / n_ch
    kl_from_uniform = np.sum(br_safe * np.log(br_safe / uniform))

    # === W decay EE ===
    ee_w = compute_w_ee(m_W, m_t)

    # === Joint EE ===
    joint_ee = ee_linear * ee_w

    # === Mass ratios ===
    mt_over_mw = m_t / m_W
    onshell_margin = (m_t - m_W - M_B) / m_t  # fraction above on-shell threshold

    # === Phase space factor for t -> bW (on-shell approximation) ===
    if m_t > m_W + M_B:
        r_W = m_W / m_t
        r_b = M_B / m_t
        # Exact 2-body phase space
        lam = (1 - (r_W + r_b)**2) * (1 - (r_W - r_b)**2)
        ps = np.sqrt(max(lam, 0)) * (1 - r_W**2 - r_b**2 +
              (r_W**2 - r_b**2)**2 / (1 - r_W**2 - r_b**2 + 1e-30)
              if abs(1 - r_W**2 - r_b**2) > 1e-10 else 0)
        # Simpler version: (1-r_W^2)^2 (1+2r_W^2)  ignoring m_b
        r = m_W / m_t
        phase_space_factor = (1 - r**2)**2 * (1 + 2 * r**2)
    else:
        phase_space_factor = 0.0

    result.update({
        "ee_linear": ee_linear,
        "ee_shannon": ee_shannon,
        "ee_renyi2": ee_renyi2,
        "n_eff": n_eff,
        "total_width": total_width,
        "width_over_mass": width_over_mass,
        "width_over_GammaW": width_over_GammaW,
        "n_channels": n_ch,
        "max_br": max_br,
        "br_ratio_12": br_ratio_12,
        "kl_from_uniform": kl_from_uniform,
        "ee_w": ee_w,
        "joint_ee": joint_ee,
        "mt_over_mw": mt_over_mw,
        "onshell_margin": onshell_margin,
        "phase_space_factor": phase_space_factor,
    })
    return result


def scan_top_yukawa(y_min=0.02, y_max=8.0, npoints=500, m_W=SM_MW):
    """Scan y_t and compute all quantities at each point.

    Returns dict of quantity_name -> array of values.
    """
    y_values = np.logspace(np.log10(y_min), np.log10(y_max), npoints)

    # Initialize storage
    all_keys = None
    results = {}

    for i, y_t in enumerate(y_values):
        q = compute_all_quantities(y_t, m_W)

        if all_keys is None:
            all_keys = list(q.keys())
            results = {k: np.zeros(npoints) for k in all_keys}

        for k in all_keys:
            results[k][i] = q[k]

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{npoints}] y_t={y_t:.4f}, m_t={q['m_t']:.1f} GeV, "
                  f"EE={q['ee_linear']:.4f}")

    # Compute numerical derivatives
    dy = np.diff(np.log(y_values))  # d(ln y)
    for base_key in ["ee_linear", "ee_shannon", "total_width"]:
        vals = results[base_key]
        # First derivative: d(Q)/d(ln y)
        dQ = np.diff(vals) / dy
        results[f"d_{base_key}"] = np.concatenate([[np.nan], dQ])
        # Second derivative
        d2Q = np.diff(dQ) / dy[1:]
        results[f"d2_{base_key}"] = np.concatenate([[np.nan, np.nan], d2Q])

    return results


def find_features(results, y_target=1.0, window=0.3):
    """Find extrema and special features near y_t = y_target.

    Returns list of (quantity_name, feature_type, y_at_feature, value) tuples.
    """
    y = results["y_t"]
    features = []

    # Quantities to check for extrema
    check_keys = [
        "ee_linear", "ee_shannon", "ee_renyi2", "n_eff",
        "width_over_mass", "width_over_GammaW",
        "max_br", "br_ratio_12", "kl_from_uniform",
        "joint_ee", "phase_space_factor",
        "d_ee_linear", "d_ee_shannon", "d_total_width",
    ]

    for key in check_keys:
        vals = results.get(key)
        if vals is None:
            continue
        valid = ~np.isnan(vals)
        if not valid.any():
            continue

        # Look for local extrema in the window around y_target
        mask = valid & (y > y_target * (1 - window)) & (y < y_target * (1 + window))
        if mask.sum() < 5:
            continue

        y_win = y[mask]
        v_win = vals[mask]

        # Check for sign change in derivative (extremum)
        dv = np.diff(v_win)
        sign_changes = np.where(np.diff(np.sign(dv)))[0]
        for sc in sign_changes:
            y_ext = y_win[sc + 1]
            v_ext = v_win[sc + 1]
            # Is it a max or min?
            if dv[sc] > 0 and dv[sc + 1] < 0:
                features.append((key, "local_max", y_ext, v_ext))
            elif dv[sc] < 0 and dv[sc + 1] > 0:
                features.append((key, "local_min", y_ext, v_ext))

        # Check for zero crossings (for derivative quantities)
        if key.startswith("d_") or key.startswith("d2_"):
            zero_crossings = np.where(np.diff(np.sign(v_win)))[0]
            for zc in zero_crossings:
                y_zc = y_win[zc + 1]
                features.append((key, "zero_crossing", y_zc, 0.0))

    # Check value at y=1
    idx_1 = np.argmin(np.abs(y - y_target))
    print(f"\n=== Values at y_t = {y[idx_1]:.4f} (closest to {y_target}) ===")
    for key in sorted(results.keys()):
        if key in ("y_t", "kappa_t"):
            continue
        val = results[key][idx_1]
        if not np.isnan(val):
            print(f"  {key:25s} = {val:.6f}")

    return features
