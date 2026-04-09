"""SM parameter defaults and decay channel definitions."""

DOCKER_IMAGE_NAME = "smparamscan-hdecay"

# SM default parameters for hdecay.in (parameter order matters)
SM_PARAMS = {
    "IHIGGS": 0,        # SM Higgs
    "TGBET": 1.0,       # tan(beta) - irrelevant for SM
    "AMABEG": 125.11,   # Higgs mass start
    "AMAEND": 125.11,   # Higgs mass end
    "NMA": 1,           # single mass point
    "ALSMZ": 0.1180,    # alpha_s(m_Z)
    "AMS": 0.100,       # m_s (MS-bar at 2 GeV)
    "AMC": 1.42,        # m_c (pole mass)
    "AMB": 4.49,        # m_b (pole mass)
    "AMT": 172.5,       # m_t (pole mass)
    "AMTAU": 1.777,     # m_tau
    "AMMUON": 0.10566,  # m_mu
    "ALPH": 137.036,    # 1/alpha_em
    "GF": 1.16637e-5,   # Fermi constant
    "GAMW": 2.085,      # W width
    "GAMZ": 2.4952,     # Z width
    "AMZ": 91.188,      # Z mass
    "AMW": 80.379,      # W mass
    "VUS": 0.2243,      # CKM
    "VCB": 0.0422,      # CKM
    "RVUB": 0.0903,     # |V_ub/V_cb|
    # SUSY params - set large to decouple
    "AMU": 1000.0,
    "AM2": 1000.0,
    "AMEL1": 1000.0,
    "AMER1": 1000.0,
    "AMQL1": 1000.0,
    "AMUR1": 1000.0,
    "AMDR1": 1000.0,
    "AMEL": 1000.0,
    "AMER": 1000.0,
    "AMSQ": 1000.0,
    "AMUR": 1000.0,
    "AMDR": 1000.0,
    "AL": 1000.0,
    "AU": 1000.0,
    "AD": 1000.0,
    # Flags
    "NNLO": 0,
    "IONSH": 0,         # include off-shell decays
    "IONWZ": 0,
    "IPOLE": 0,
    "IOFSUSY": 0,
    "INDIDEC": 0,
    "NFGG": 5,
}

# Ordered list of parameter names matching hdecay.in read order
PARAM_ORDER = [
    "IHIGGS", "TGBET", "AMABEG", "AMAEND", "NMA",
    "ALSMZ", "AMS", "AMC", "AMB", "AMT",
    "AMTAU", "AMMUON", "ALPH", "GF", "GAMW",
    "GAMZ", "AMZ", "AMW", "VUS", "VCB", "RVUB",
    "AMU", "AM2", "AMEL1", "AMER1", "AMQL1",
    "AMUR1", "AMDR1", "AMEL", "AMER", "AMSQ",
    "AMUR", "AMDR", "AL", "AU", "AD",
    "NNLO", "IONSH", "IONWZ", "IPOLE", "IOFSUSY",
    "INDIDEC", "NFGG",
]

# Integer-valued parameters (use FORMAT 101)
INT_PARAMS = {"IHIGGS", "NMA", "NNLO", "IONSH", "IONWZ", "IPOLE",
              "IOFSUSY", "INDIDEC", "NFGG"}

# Mapping: quark name -> HDECAY mass parameter
QUARK_MASS_PARAMS = {
    "u": None,   # HDECAY doesn't have a direct u-mass input; u is effectively massless
    "d": None,   # Same for d
    "s": "AMS",
    "c": "AMC",
    "b": "AMB",
    "t": "AMT",
}

# For u and d quarks, we'll note that HDECAY treats them as massless internally.
# Varying their Yukawa has no effect in HDECAY since h->uu and h->dd BRs are
# computed from running masses which are negligibly small. We'll still run the
# scan but the EE will be essentially flat.

# br.sm1 columns (after MHSM): BB, TAU TAU, MU MU, SS, CC, TT
BR_SM1_CHANNELS = ["bb", "tautau", "mumu", "ss", "cc", "tt"]

# br.sm2 columns (after MHSM): GG, GAM GAM, Z GAM, WW, ZZ, WIDTH
BR_SM2_CHANNELS = ["gg", "gamgam", "zgam", "WW", "ZZ"]
# WIDTH is the last column but not a BR

# All decay channels used in EE calculation
ALL_CHANNELS = BR_SM1_CHANNELS + BR_SM2_CHANNELS

# Spin factors P_i for each channel
# Fermions: P = 1/2; Bosons gg/gamgam/zgam: P = 1/2; WW/ZZ: computed from kinematics
SPIN_FACTORS = {
    "bb": 0.5, "tautau": 0.5, "mumu": 0.5,
    "ss": 0.5, "cc": 0.5, "tt": 0.5,
    "gg": 0.5, "gamgam": 0.5, "zgam": 0.5,
    # WW and ZZ set dynamically via three-body computation
    "WW": None,
    "ZZ": None,
}

# Color multiplicity N_c for each channel
COLOR_FACTORS = {
    "bb": 3, "cc": 3, "ss": 3, "tt": 3,  # quarks
    "tautau": 1, "mumu": 1,                # leptons
    "gg": 8,                                # gluons
    "gamgam": 1, "zgam": 1,                 # photons
    "WW": 1, "ZZ": 1,                       # weak bosons
}

# Physical masses for spin factor computation
M_HIGGS = 125.11  # GeV
M_W = 80.379      # GeV
M_Z = 91.188      # GeV

# Quark labels for plotting
QUARK_LABELS = {
    "u": r"$\kappa_u$",
    "d": r"$\kappa_d$",
    "s": r"$\kappa_s$",
    "c": r"$\kappa_c$",
    "b": r"$\kappa_b$",
    "t": r"$\kappa_t$",
}

QUARK_NAMES = {
    "u": "up", "d": "down", "s": "strange",
    "c": "charm", "b": "bottom", "t": "top",
}
