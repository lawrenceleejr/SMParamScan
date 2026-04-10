"""Analytical Higgs branching ratios and EE as a function of y_t (top Yukawa).

Varies the top quark mass m_t = y_t * v / sqrt(2) while keeping all other SM
parameters fixed. Loop-induced channels (h->gg, h->gamgam) are rescaled using
form factor ratios. The h->tt tree-level channel opens when m_t < m_H/2.

Form factors follow the conventions of Djouadi (hep-ph/0503173):
  tau = m_H^2 / (4 m^2)
  A_{1/2}(tau) = 2[tau + (tau-1)f(tau)] / tau^2       (fermion loop)
  A_1(tau) = -[2 tau^2 + 3 tau + 3(2tau-1) f(tau)] / tau^2  (W loop)
"""

import numpy as np
from scipy import integrate

from .entropy import P_WW, P_ZZ

# ── Physical constants ──────────────────────────────────────────────
V_HIGGS = 246.22   # Higgs vev [GeV]
M_HIGGS = 125.11   # Higgs mass [GeV]
M_W = 80.379       # W mass [GeV]
M_Z = 91.188       # Z mass [GeV]
G_F = 1.16637e-5   # Fermi constant [GeV^-2]
SM_MT = 172.5       # SM top pole mass [GeV]
SM_MB = 4.49        # b pole mass [GeV]
SM_MC = 1.42        # c pole mass [GeV]
SM_MTAU = 1.777     # tau mass [GeV]

# ── SM branching ratios at m_H = 125.11 GeV ────────────────────────
# From LHC HXSWG YR4 (HDECAY-based)
SM_TOTAL_WIDTH = 4.10e-3  # GeV (4.10 MeV)
SM_BRS = {
    "bb":     0.5824,
    "WW":     0.2137,
    "gg":     0.0819,
    "tautau": 0.0627,
    "cc":     0.0288,
    "ZZ":     0.0262,
    "gamgam": 0.00227,
    "zgam":   0.00154,
    "mumu":   0.000218,
    "ss":     0.000240,
    "tt":     0.0,
}
SM_PARTIAL_WIDTHS = {ch: br * SM_TOTAL_WIDTH for ch, br in SM_BRS.items()}

# Spin and color factors from arXiv:2511.17321
SPIN_FACTORS = {
    "bb": 0.5, "cc": 0.5, "ss": 0.5, "tt": 0.5,
    "tautau": 0.5, "mumu": 0.5,
    "gg": 0.5, "gamgam": 0.5, "zgam": 0.5,
    "WW": P_WW, "ZZ": P_ZZ,
}
COLOR_FACTORS = {
    "bb": 3, "cc": 3, "ss": 3, "tt": 3,
    "tautau": 1, "mumu": 1,
    "gg": 8, "gamgam": 1, "zgam": 1,
    "WW": 1, "ZZ": 1,
}

ALL_CHANNELS = list(SM_BRS.keys())


# ── Loop form factors ───────────────────────────────────────────────

def _f_tau(tau):
    """f(tau) for loop form factors.

    tau = m_H^2 / (4 m^2).
    tau <= 1: heavy particle (below threshold), f is real.
    tau > 1:  light particle (above threshold), f is complex.
    """
    if tau <= 1.0:
        return complex(np.arcsin(np.sqrt(tau))**2)
    else:
        beta = np.sqrt(1.0 - 1.0 / tau)
        log_arg = (1.0 + beta) / (1.0 - beta)
        return -0.25 * (np.log(log_arg) - 1j * np.pi)**2


def _A_half(tau):
    """Fermion loop form factor A_{1/2}(tau).

    Heavy limit (tau -> 0): A_{1/2} -> 4/3.
    Light limit (tau -> inf): A_{1/2} -> 0.
    """
    f = _f_tau(tau)
    return 2.0 * (tau + (tau - 1.0) * f) / tau**2


def _A_one(tau):
    """W boson loop form factor A_1(tau).

    Heavy limit (tau -> 0): A_1 -> -7.
    """
    f = _f_tau(tau)
    return -(2.0 * tau**2 + 3.0 * tau + 3.0 * (2.0 * tau - 1.0) * f) / tau**2


def _tau(m_H, m):
    """tau = m_H^2 / (4 m^2)."""
    return m_H**2 / (4.0 * m**2)


# ── Amplitude computations ──────────────────────────────────────────

def _gg_amplitude_sq(m_t):
    """| Sum_q A_{1/2}(tau_q) |^2  for h -> gg."""
    A = complex(0)
    for m_q in [m_t, SM_MB, SM_MC]:
        if m_q > 0.01:
            A += _A_half(_tau(M_HIGGS, m_q))
    return abs(A)**2


def _gamgam_amplitude_sq(m_t):
    """| Sum_f Nc Qf^2 A_{1/2} + A_1(tau_W) |^2  for h -> gamgam."""
    A = complex(0)
    # Quarks: top, bottom, charm (others negligible)
    quarks = [(m_t, 2.0/3.0), (SM_MB, -1.0/3.0), (SM_MC, 2.0/3.0)]
    for m_q, Q in quarks:
        if m_q > 0.01:
            A += 3.0 * Q**2 * _A_half(_tau(M_HIGGS, m_q))
    # Tau lepton
    A += 1.0 * 1.0**2 * _A_half(_tau(M_HIGGS, SM_MTAU))
    # W boson
    A += _A_one(_tau(M_HIGGS, M_W))
    return abs(A)**2


# SM reference values (computed at import time)
_GG_SM = _gg_amplitude_sq(SM_MT)
_GAMGAM_SM = _gamgam_amplitude_sq(SM_MT)


def _htt_partial_width(m_t):
    """h -> tt-bar tree-level partial width [GeV].

    Gamma = N_c G_F m_t^2 m_H / (4 sqrt(2) pi) * beta^3
    where beta = sqrt(1 - 4 m_t^2 / m_H^2).
    """
    if m_t <= 0 or M_HIGGS <= 2.0 * m_t:
        return 0.0
    beta_sq = 1.0 - (2.0 * m_t / M_HIGGS)**2
    beta = np.sqrt(beta_sq)
    return 3.0 * G_F * m_t**2 * M_HIGGS / (4.0 * np.sqrt(2) * np.pi) * beta**3


# ── Main computation ────────────────────────────────────────────────

def compute_higgs_brs(y_t):
    """Compute all Higgs BRs as a function of y_t.

    Returns (brs_dict, total_width_GeV) or (None, 0) if no channels.
    """
    m_t = y_t * V_HIGGS / np.sqrt(2)

    widths = dict(SM_PARTIAL_WIDTHS)

    # Scale loop-induced channels
    R_gg = _gg_amplitude_sq(m_t) / _GG_SM
    widths["gg"] = SM_PARTIAL_WIDTHS["gg"] * R_gg

    R_gam = _gamgam_amplitude_sq(m_t) / _GAMGAM_SM
    widths["gamgam"] = SM_PARTIAL_WIDTHS["gamgam"] * R_gam

    # h -> Zgamma: top contributes ~10% of amplitude.
    # Approximate: scale Zgam similarly to gamgam (small channel, 0.15% BR).
    widths["zgam"] = SM_PARTIAL_WIDTHS["zgam"] * R_gam

    # h -> tt-bar tree-level
    widths["tt"] = _htt_partial_width(m_t)

    total = sum(widths.values())
    if total <= 0:
        return None, 0.0

    brs = {ch: w / total for ch, w in widths.items()}
    return brs, total


def compute_higgs_ee_partonic(brs):
    """Partonic Higgs EE using paper's formula with spin/color factors.

    EE = 1 - Sum_i (P_i / N_c^i) BR_i^2
    """
    purity = 0.0
    for ch in ALL_CHANNELS:
        br = brs.get(ch, 0.0)
        P = SPIN_FACTORS[ch]
        Nc = COLOR_FACTORS[ch]
        purity += (P / Nc) * br**2
    return 1.0 - purity


def compute_higgs_ee_simple(brs):
    """Simple linear entropy: EE = 1 - Sum BR_i^2."""
    return 1.0 - sum(br**2 for br in brs.values())


def compute_higgs_shannon(brs):
    """Shannon entropy: -Sum BR_i ln(BR_i)."""
    S = 0.0
    for br in brs.values():
        if br > 0:
            S -= br * np.log(br)
    return S


def compute_ee_max(channels):
    """Maximum achievable EE for a set of channels.

    Maximizes EE = 1 - Sum (P_i/N_c^i) BR_i^2 subject to Sum BR_i = 1.
    Solution: BR_i = (N_c^i / P_i) / Sum_j (N_c^j / P_j).
    EE_max = 1 - 1 / Sum_j (N_c^j / P_j).
    """
    inv_weights = 0.0
    for ch in channels:
        P = SPIN_FACTORS[ch]
        Nc = COLOR_FACTORS[ch]
        inv_weights += Nc / P
    if inv_weights <= 0:
        return 0.0
    return 1.0 - 1.0 / inv_weights


# Pre-compute EE_max for 10 channels (no tt) and 11 channels (with tt)
_CHANNELS_NO_TT = [ch for ch in ALL_CHANNELS if ch != "tt"]
_CHANNELS_WITH_TT = list(ALL_CHANNELS)
EE_MAX_NO_TT = compute_ee_max(_CHANNELS_NO_TT)
EE_MAX_WITH_TT = compute_ee_max(_CHANNELS_WITH_TT)


def compute_all_higgs_quantities(y_t):
    """Compute all interesting Higgs quantities at a given y_t.

    Returns dict of quantity_name -> value.
    """
    m_t = y_t * V_HIGGS / np.sqrt(2)

    result = {
        "y_t": y_t,
        "m_t": m_t,
        "kappa_t": m_t / SM_MT,
    }

    brs_result = compute_higgs_brs(y_t)
    brs, total_width = brs_result

    if brs is None:
        for key in ["ee_partonic", "ee_simple", "shannon", "total_width",
                     "R_gg", "R_gamgam", "br_bb", "br_WW", "br_gg",
                     "br_gamgam", "br_tt", "br_ZZ", "br_tautau", "br_cc",
                     "width_ratio", "ee_max", "ee_frac", "purity_bb",
                     "purity_gg", "gamgam_signal_strength"]:
            result[key] = np.nan
        return result

    # EE variants
    ee_part = compute_higgs_ee_partonic(brs)
    result["ee_partonic"] = ee_part
    result["ee_simple"] = compute_higgs_ee_simple(brs)
    result["shannon"] = compute_higgs_shannon(brs)

    # Maximum achievable EE (depends on which channels are open)
    tt_open = brs.get("tt", 0.0) > 1e-10
    ee_max = EE_MAX_WITH_TT if tt_open else EE_MAX_NO_TT
    result["ee_max"] = ee_max
    result["ee_frac"] = ee_part / ee_max  # fraction of maximum

    # Width
    result["total_width"] = total_width
    result["width_ratio"] = total_width / SM_TOTAL_WIDTH

    # Loop amplitude ratios
    R_gg = _gg_amplitude_sq(m_t) / _GG_SM
    R_gam = _gamgam_amplitude_sq(m_t) / _GAMGAM_SM
    result["R_gg"] = R_gg
    result["R_gamgam"] = R_gam

    # Signal strengths: mu_xx = R_gg * BR_xx / BR_xx_SM
    # (production via gluon fusion × decay BR, normalized to SM)
    for ch in ["gamgam", "WW", "ZZ", "bb", "tautau"]:
        br_sm = SM_BRS.get(ch, 0.0)
        if br_sm > 0:
            result[f"mu_{ch}"] = R_gg * brs[ch] / br_sm
        else:
            result[f"mu_{ch}"] = np.nan

    # gamgam signal strength (special: involves both production and decay loops)
    result["gamgam_signal_strength"] = R_gg * R_gam * SM_TOTAL_WIDTH / total_width

    # Individual BRs
    for ch in ALL_CHANNELS:
        result[f"br_{ch}"] = brs.get(ch, 0.0)

    # Purity breakdown
    purity_gg = (SPIN_FACTORS["gg"] / COLOR_FACTORS["gg"]) * brs["gg"]**2
    purity_bb = (SPIN_FACTORS["bb"] / COLOR_FACTORS["bb"]) * brs["bb"]**2
    result["purity_gg"] = purity_gg
    result["purity_bb"] = purity_bb

    # Interference measure: top contribution fraction in gamgam amplitude
    A_gamgam_full_sq = _gamgam_amplitude_sq(m_t)
    # Compute without top
    A_no_top = complex(0)
    A_no_top += 3.0 * (1.0/3.0)**2 * _A_half(_tau(M_HIGGS, SM_MB))  # b
    A_no_top += 3.0 * (2.0/3.0)**2 * _A_half(_tau(M_HIGGS, SM_MC))  # c
    A_no_top += 1.0 * _A_half(_tau(M_HIGGS, SM_MTAU))  # tau
    A_no_top += _A_one(_tau(M_HIGGS, M_W))  # W
    R_gamgam_no_top = abs(A_no_top)**2
    # Interference fraction: how much the top changes |A|^2
    result["gamgam_interf"] = 1.0 - A_gamgam_full_sq / R_gamgam_no_top

    return result


def _gamgam_amplitude_sq_kf(kappa_f):
    """h->gamgam with universal kappa_f: coupling scales, masses stay at SM."""
    A = complex(0)
    # Fermion loops at SM masses, but coupling scaled by kappa_f
    quarks = [(SM_MT, 2.0/3.0), (SM_MB, -1.0/3.0), (SM_MC, 2.0/3.0)]
    for m_q, Q in quarks:
        A += kappa_f * 3.0 * Q**2 * _A_half(_tau(M_HIGGS, m_q))
    A += kappa_f * 1.0 * _A_half(_tau(M_HIGGS, SM_MTAU))  # tau
    # W boson loop (unscaled)
    A += _A_one(_tau(M_HIGGS, M_W))
    return abs(A)**2


# SM reference for kf scaling
_GAMGAM_SM_KF = _gamgam_amplitude_sq_kf(1.0)
_GG_FERM_SUM_SM = abs(
    _A_half(_tau(M_HIGGS, SM_MT))
    + _A_half(_tau(M_HIGGS, SM_MB))
    + _A_half(_tau(M_HIGGS, SM_MC))
)**2


def compute_higgs_brs_kf(kappa_f):
    """Compute Higgs BRs under universal kappa_f rescaling (paper's approach).

    All fermion Yukawa couplings are scaled by kappa_f; fermion masses stay at
    SM values for kinematics and form factors. This modifies:
      - Tree-level fermion widths: Gamma_ff = kf^2 * Gamma_ff^SM
      - h->gg: Gamma_gg = kf^2 * Gamma_gg^SM (all loop couplings scale)
      - h->gamgam: interference between kf-scaled fermion loops and W loop
      - h->WW, h->ZZ: unchanged
    """
    kf2 = kappa_f**2

    widths = {}
    # Tree-level fermion channels scale as kf^2
    for ch in ["bb", "tautau", "cc", "ss", "mumu", "tt"]:
        widths[ch] = kf2 * SM_PARTIAL_WIDTHS[ch]

    # h->gg: all quark loop couplings scale by kf
    widths["gg"] = kf2 * SM_PARTIAL_WIDTHS["gg"]

    # h->gamgam: fermion loops scale by kf, W loop doesn't
    R_gam_kf = _gamgam_amplitude_sq_kf(kappa_f) / _GAMGAM_SM_KF
    widths["gamgam"] = SM_PARTIAL_WIDTHS["gamgam"] * R_gam_kf

    # h->Zgam: similar interference, approximate same as gamgam
    widths["zgam"] = SM_PARTIAL_WIDTHS["zgam"] * R_gam_kf

    # Vector boson channels: unchanged
    widths["WW"] = SM_PARTIAL_WIDTHS["WW"]
    widths["ZZ"] = SM_PARTIAL_WIDTHS["ZZ"]

    total = sum(widths.values())
    if total <= 0:
        return None, 0.0
    brs = {ch: w / total for ch, w in widths.items()}
    return brs, total


def compute_all_kf_quantities(kappa_f):
    """Compute Higgs EE and related quantities under universal kappa_f."""
    result = {"kappa_f": kappa_f}

    brs, total_width = compute_higgs_brs_kf(kappa_f)
    if brs is None:
        for key in ["ee_partonic", "ee_simple", "shannon", "total_width",
                     "width_ratio", "ee_frac"]:
            result[key] = np.nan
        return result

    result["ee_partonic"] = compute_higgs_ee_partonic(brs)
    result["ee_simple"] = compute_higgs_ee_simple(brs)
    result["shannon"] = compute_higgs_shannon(brs)
    result["total_width"] = total_width
    result["width_ratio"] = total_width / SM_TOTAL_WIDTH
    result["ee_frac"] = result["ee_partonic"] / EE_MAX_NO_TT

    # Key BRs
    for ch in ALL_CHANNELS:
        result[f"br_{ch}"] = brs.get(ch, 0.0)

    # gamgam signal strength: sigma(gg->h)*BR(gamgam) / SM
    # production scales as kf^2 (gg->h), BR_gamgam = Gamma_gamgam/Gamma_total
    R_gam_kf = _gamgam_amplitude_sq_kf(kappa_f) / _GAMGAM_SM_KF
    result["mu_gamgam"] = kappa_f**2 * R_gam_kf * SM_TOTAL_WIDTH / total_width

    return result


def scan_higgs_kf(kf_min=0.05, kf_max=10.0, npoints=500):
    """Scan universal kappa_f and compute all quantities.

    Returns dict of quantity_name -> array.
    """
    kf_values = np.logspace(np.log10(kf_min), np.log10(kf_max), npoints)

    all_keys = None
    results = {}

    for i, kf in enumerate(kf_values):
        q = compute_all_kf_quantities(kf)

        if all_keys is None:
            all_keys = list(q.keys())
            results = {k: np.zeros(npoints) for k in all_keys}

        for k in all_keys:
            results[k][i] = q[k]

        if (i + 1) % 100 == 0 or i == 0:
            ee = q.get("ee_partonic", np.nan)
            print(f"  [{i+1}/{npoints}] kf={kf:.4f}, "
                  f"EE={ee:.6f}, EE/max={q.get('ee_frac', np.nan):.6f}")

    # Numerical derivatives
    dk = np.diff(np.log(kf_values))
    for base_key in ["ee_partonic", "ee_simple", "shannon"]:
        vals = results.get(base_key)
        if vals is None:
            continue
        dQ = np.diff(vals) / dk
        results[f"d_{base_key}"] = np.concatenate([[np.nan], dQ])
        d2Q = np.diff(dQ) / dk[1:]
        results[f"d2_{base_key}"] = np.concatenate([[np.nan, np.nan], d2Q])

    return results


def scan_higgs_ee(y_min=0.02, y_max=10.0, npoints=500):
    """Scan y_t and compute all Higgs quantities.

    Returns dict of quantity_name -> array.
    """
    y_values = np.logspace(np.log10(y_min), np.log10(y_max), npoints)

    all_keys = None
    results = {}

    for i, y_t in enumerate(y_values):
        q = compute_all_higgs_quantities(y_t)

        if all_keys is None:
            all_keys = list(q.keys())
            results = {k: np.zeros(npoints) for k in all_keys}

        for k in all_keys:
            results[k][i] = q[k]

        if (i + 1) % 100 == 0 or i == 0:
            ee = q.get("ee_partonic", np.nan)
            print(f"  [{i+1}/{npoints}] y_t={y_t:.4f}, m_t={q['m_t']:.1f} GeV, "
                  f"EE_partonic={ee:.6f}")

    # Numerical derivatives wrt ln(y_t)
    dy = np.diff(np.log(y_values))
    for base_key in ["ee_partonic", "ee_simple", "shannon", "total_width"]:
        vals = results.get(base_key)
        if vals is None:
            continue
        dQ = np.diff(vals) / dy
        results[f"d_{base_key}"] = np.concatenate([[np.nan], dQ])
        d2Q = np.diff(dQ) / dy[1:]
        results[f"d2_{base_key}"] = np.concatenate([[np.nan, np.nan], d2Q])

    return results


def find_higgs_features(results, y_target=1.0, window=0.5):
    """Find extrema and special features near y_t = y_target.

    Returns list of (quantity_name, feature_type, y_at_feature, value).
    """
    y = results["y_t"]
    features = []

    check_keys = [
        "ee_partonic", "ee_simple", "shannon",
        "total_width", "width_ratio",
        "R_gg", "R_gamgam",
        "br_bb", "br_WW", "br_gg", "br_gamgam", "br_tt",
        "d_ee_partonic", "d_ee_simple", "d_shannon",
    ]

    for key in check_keys:
        vals = results.get(key)
        if vals is None:
            continue
        valid = ~np.isnan(vals)
        if not valid.any():
            continue

        mask = valid & (y > y_target * (1 - window)) & (y < y_target * (1 + window))
        if mask.sum() < 5:
            continue

        y_win = y[mask]
        v_win = vals[mask]

        # Local extrema
        dv = np.diff(v_win)
        sign_changes = np.where(np.diff(np.sign(dv)))[0]
        for sc in sign_changes:
            y_ext = y_win[sc + 1]
            v_ext = v_win[sc + 1]
            if dv[sc] > 0 and dv[sc + 1] < 0:
                features.append((key, "local_max", float(y_ext), float(v_ext)))
            elif dv[sc] < 0 and dv[sc + 1] > 0:
                features.append((key, "local_min", float(y_ext), float(v_ext)))

        # Zero crossings (for derivatives)
        if key.startswith("d_"):
            zero_crossings = np.where(np.diff(np.sign(v_win)))[0]
            for zc in zero_crossings:
                features.append((key, "zero_crossing", float(y_win[zc + 1]), 0.0))

    # Print values at y_t = target
    idx = np.argmin(np.abs(y - y_target))
    print(f"\n=== Higgs quantities at y_t = {y[idx]:.4f} ===")
    for key in sorted(results.keys()):
        if key in ("y_t", "kappa_t", "m_t"):
            continue
        val = results[key][idx]
        if not np.isnan(val):
            print(f"  {key:25s} = {val:.8f}")

    return features
