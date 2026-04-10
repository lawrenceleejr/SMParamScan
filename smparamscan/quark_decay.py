"""Compute quark weak-decay branching ratios and entanglement entropy.

Each quark decays via q -> q' + W* -> q' + ff', where the W* can produce
leptons (l nu) or quark pairs (q1 q2). The branching ratios depend on the
parent quark mass through phase space and the W propagator.

As the quark mass increases:
  - More channels open (phase space thresholds)
  - The W propagator enhances the rate near m_W
  - Above m_W + m_q', the W goes on-shell
"""

import numpy as np
from scipy import integrate

# Physical constants
G_F = 1.16637e-5   # GeV^-2
M_W = 80.379        # GeV
GAMMA_W = 2.085     # GeV

# Quark masses (GeV) — used for final-state phase space
QUARK_MASSES = {
    "u": 0.0022, "d": 0.0047, "s": 0.095, "c": 1.27, "b": 4.18, "t": 172.5,
}

# Lepton masses (GeV)
LEPTON_MASSES = {"e": 0.000511, "mu": 0.1057, "tau": 1.777, "nu": 0.0}

# CKM matrix |V_ij|^2 (i=up-type, j=down-type)
CKM2 = {
    ("u", "d"): 0.974**2, ("u", "s"): 0.225**2, ("u", "b"): 0.00365**2,
    ("c", "d"): 0.225**2, ("c", "s"): 0.974**2, ("c", "b"): 0.0412**2,
    ("t", "d"): 0.009**2, ("t", "s"): 0.040**2, ("t", "b"): 0.999**2,
}

# Up-type quarks (decay to down-type + W+)
UP_TYPE = {"u", "c", "t"}
# Down-type quarks (decay to up-type + W-)
DOWN_TYPE = {"d", "s", "b"}

# W* leptonic decay modes: (lepton, neutrino)
W_LEPTONIC = [
    ("e", LEPTON_MASSES["e"], LEPTON_MASSES["nu"]),
    ("mu", LEPTON_MASSES["mu"], LEPTON_MASSES["nu"]),
    ("tau", LEPTON_MASSES["tau"], LEPTON_MASSES["nu"]),
]

# W* hadronic decay modes: (label, quark1, quark2, |V|^2, Nc)
# W- -> u_bar d_j  (CKM: V_ud, V_us, V_cd, V_cs, etc.)
# We list all up-down pairs with their CKM factors
W_HADRONIC_PAIRS = [
    ("ud", "u", "d"), ("us", "u", "s"),
    ("cd", "c", "d"), ("cs", "c", "s"),
]


def _breit_wigner(s):
    """W propagator squared, normalized to 1 in the Fermi limit."""
    return M_W**4 / ((s - M_W**2)**2 + M_W**2 * GAMMA_W**2)


def _dalitz_s2_limits(s1, M, m1, m2, m3):
    """Compute s2 limits for the Dalitz plot at fixed s1.

    q(M) -> 1(m1) + 2(m2) + 3(m3), s1 = (p2+p3)^2, s2 = (p1+p3)^2.
    Energies computed in the s1 rest frame.
    """
    sqrt_s1 = np.sqrt(s1)
    # Energy of particle 1 in the (2,3) rest frame
    E1 = (M**2 - m1**2 - s1) / (2 * sqrt_s1)
    # Energy of particle 3 in the (2,3) rest frame
    E3 = (s1 + m3**2 - m2**2) / (2 * sqrt_s1)

    if E1 < m1 or E3 < m3:
        return None, None

    p1 = np.sqrt(max(E1**2 - m1**2, 0))
    p3 = np.sqrt(max(E3**2 - m3**2, 0))

    s2_max = (E1 + E3)**2 - (p1 - p3)**2
    s2_min = (E1 + E3)**2 - (p1 + p3)**2

    if s2_min > s2_max:
        return None, None

    return s2_min, s2_max


def _s2_integral(s2_min, s2_max, M, m1, m2, m3):
    """Analytically integrate (M^2+m2^2-s2)(s2-m1^2-m3^2) over s2."""
    a = M**2 + m2**2
    b = m1**2 + m3**2
    # Integral of -(s^2) + (a+b)s - ab from s_min to s_max
    def F(s):
        return -s**3 / 3 + (a + b) * s**2 / 2 - a * b * s
    return F(s2_max) - F(s2_min)


def partial_width(M_parent, m_daughter, m_f1, m_f2, V_prod_sq, V_decay_sq, Nc):
    """Compute partial width for q(M) -> q'(m_daughter) + f1(m_f1) + f2(m_f2).

    Uses the full W propagator (Breit-Wigner), valid for all quark masses
    from light quarks (Fermi theory) through the top (on-shell W).

    Gamma = G_F^2 |V_prod|^2 |V_decay|^2 Nc / (16 pi^3 M^3) *
            integral over Dalitz plot of BW(s1) * (M^2+m2^2-s2)(s2-m1^2-m3^2)
    """
    if M_parent <= m_daughter + m_f1 + m_f2:
        return 0.0  # kinematically forbidden

    s1_min = (m_f1 + m_f2)**2
    s1_max = (M_parent - m_daughter)**2

    def integrand(s1):
        s2_min, s2_max = _dalitz_s2_limits(s1, M_parent, m_daughter, m_f1, m_f2)
        if s2_min is None:
            return 0.0
        I_s2 = _s2_integral(s2_min, s2_max, M_parent, m_daughter, m_f1, m_f2)
        return _breit_wigner(s1) * I_s2

    result, _ = integrate.quad(integrand, s1_min, s1_max, limit=200)
    prefactor = G_F**2 * V_prod_sq * V_decay_sq * Nc / (16 * np.pi**3 * M_parent**3)
    return prefactor * result


def get_decay_channels(quark, quark_mass):
    """Return all kinematically allowed decay channels with partial widths.

    Returns list of (label, partial_width) tuples.
    """
    channels = []

    if quark in UP_TYPE:
        # Up-type: q -> d_j + W+* -> d_j + (f fbar')
        daughters = ["d", "s", "b"]
        for dq in daughters:
            m_dq = QUARK_MASSES[dq]
            V_prod_sq = CKM2.get((quark, dq), 0.0)
            if V_prod_sq == 0:
                continue

            # Leptonic modes
            for lep_name, m_l, m_nu in W_LEPTONIC:
                pw = partial_width(quark_mass, m_dq, m_l, m_nu, V_prod_sq, 1.0, 1)
                if pw > 0:
                    channels.append((f"{dq}+{lep_name}nu", pw))

            # Hadronic modes (W+ -> u_i dbar_j)
            for hlabel, q1, q2 in W_HADRONIC_PAIRS:
                m_q1 = QUARK_MASSES[q1]
                m_q2 = QUARK_MASSES[q2]
                V_W_sq = CKM2.get((q1, q2), 0.0)
                pw = partial_width(quark_mass, m_dq, m_q1, m_q2,
                                   V_prod_sq, V_W_sq, 3)
                if pw > 0:
                    channels.append((f"{dq}+{hlabel}", pw))

    elif quark in DOWN_TYPE:
        # Down-type: q -> u_i + W-* -> u_i + (f fbar')
        daughters = ["u", "c", "t"]
        for uq in daughters:
            m_uq = QUARK_MASSES[uq]
            V_prod_sq = CKM2.get((uq, quark), 0.0)
            if V_prod_sq == 0:
                continue

            # Leptonic modes
            for lep_name, m_l, m_nu in W_LEPTONIC:
                pw = partial_width(quark_mass, m_uq, m_l, m_nu, V_prod_sq, 1.0, 1)
                if pw > 0:
                    channels.append((f"{uq}+{lep_name}nu", pw))

            # Hadronic modes (W- -> ubar_i d_j)
            for hlabel, q1, q2 in W_HADRONIC_PAIRS:
                m_q1 = QUARK_MASSES[q1]
                m_q2 = QUARK_MASSES[q2]
                V_W_sq = CKM2.get((q1, q2), 0.0)
                pw = partial_width(quark_mass, m_uq, m_q1, m_q2,
                                   V_prod_sq, V_W_sq, 3)
                if pw > 0:
                    channels.append((f"{uq}+{hlabel}", pw))

    return channels


def compute_quark_decay_brs(quark, quark_mass):
    """Compute branching ratios for all decay channels of a quark.

    Returns:
        Dict mapping channel label -> BR, or None if quark is stable.
    """
    channels = get_decay_channels(quark, quark_mass)
    if not channels:
        return None

    total = sum(pw for _, pw in channels)
    if total <= 0:
        return None

    return {label: pw / total for label, pw in channels}


def compute_quark_ee(branching_ratios):
    """Compute EE from quark decay branching ratios.

    Uses the simple linear entropy: EE = 1 - sum BR_i^2
    This measures how democratically the quark decays across channels.
    """
    if branching_ratios is None:
        return np.nan
    return 1.0 - sum(br**2 for br in branching_ratios.values())
