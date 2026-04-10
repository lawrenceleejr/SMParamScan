"""W boson decay branching ratios and entanglement entropy.

Computes W -> f1 f2bar (2-body) partial widths, branching ratios, and EE
as a function of m_W. Channels include leptonic (l nu) and hadronic (qq')
modes with CKM matrix elements and Nc=3 color factor.
"""

import numpy as np

G_F = 1.16637e-5  # GeV^-2

QUARK_MASSES = {
    "u": 0.0022, "d": 0.0047, "s": 0.095, "c": 1.27, "b": 4.18, "t": 172.5,
}
LEPTON_MASSES = {"e": 0.000511, "mu": 0.1057, "tau": 1.777}

CKM2 = {
    ("u", "d"): 0.974**2, ("u", "s"): 0.225**2, ("u", "b"): 0.00365**2,
    ("c", "d"): 0.225**2, ("c", "s"): 0.974**2, ("c", "b"): 0.0412**2,
    ("t", "d"): 0.009**2, ("t", "s"): 0.040**2, ("t", "b"): 0.999**2,
}

SM_MW = 80.379
SM_MT = 172.5


def w_partial_width(m_W, m1, m2, V_sq, Nc):
    """Partial width for W -> f1 f2bar (2-body).

    Gamma = G_F m_W^3 / (6 pi sqrt(2)) * |V|^2 * Nc * f(m1, m2)
    where f includes phase-space corrections for massive final states.
    """
    if m_W <= m1 + m2:
        return 0.0
    x1sq = (m1 / m_W)**2
    x2sq = (m2 / m_W)**2
    lam = (1 - x1sq - x2sq)**2 - 4 * x1sq * x2sq
    if lam <= 0:
        return 0.0
    sqrt_lam = np.sqrt(lam)
    ps = sqrt_lam * (1 - (x1sq + x2sq) / 2 - (x1sq - x2sq)**2 / 2)
    return G_F * m_W**3 / (6 * np.pi * np.sqrt(2)) * V_sq * Nc * ps


def get_w_decay_channels(m_W, m_t=SM_MT):
    """Return (label, partial_width) for all kinematically allowed W decays."""
    channels = []

    # Leptonic: W -> l nu_l
    for name, m_l in LEPTON_MASSES.items():
        pw = w_partial_width(m_W, m_l, 0.0, 1.0, 1)
        if pw > 0:
            channels.append((f"{name}nu", pw))

    # Hadronic: W -> u_i dbar_j (all CKM combinations)
    up_quarks = [("u", QUARK_MASSES["u"]), ("c", QUARK_MASSES["c"]), ("t", m_t)]
    down_quarks = [("d", QUARK_MASSES["d"]), ("s", QUARK_MASSES["s"]),
                   ("b", QUARK_MASSES["b"])]

    for uname, m_u in up_quarks:
        for dname, m_d in down_quarks:
            V_sq = CKM2.get((uname, dname), 0.0)
            if V_sq == 0:
                continue
            pw = w_partial_width(m_W, m_u, m_d, V_sq, 3)
            if pw > 0:
                channels.append((f"{uname}{dname}", pw))

    return channels


def compute_w_total_width(m_W, m_t=SM_MT):
    """Total W width at given m_W (and m_t for tb threshold)."""
    return sum(pw for _, pw in get_w_decay_channels(m_W, m_t))


def compute_w_brs(m_W, m_t=SM_MT):
    """W decay branching ratios."""
    channels = get_w_decay_channels(m_W, m_t)
    if not channels:
        return None
    total = sum(pw for _, pw in channels)
    if total <= 0:
        return None
    return {label: pw / total for label, pw in channels}


def compute_w_ee(m_W, m_t=SM_MT):
    """W decay entanglement entropy: EE = 1 - sum BR_i^2."""
    brs = compute_w_brs(m_W, m_t)
    if brs is None:
        return np.nan
    return 1.0 - sum(br**2 for br in brs.values())
