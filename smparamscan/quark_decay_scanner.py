"""Scan quark Yukawa couplings and compute EE from quark weak decays."""

import numpy as np

from .quark_decay import (
    QUARK_MASSES,
    compute_quark_decay_brs,
    compute_quark_ee,
)


def scan_quark_decay(
    quark: str,
    kappa_min: float = 0.01,
    kappa_max: float = 100.0,
    npoints: int = 200,
) -> tuple[np.ndarray, np.ndarray, list[dict | None]]:
    """Scan a single quark's weak-decay EE as a function of kappa.

    Returns:
        Tuple of (kappa_values, ee_values, br_list) where br_list[i] is
        the BR dict at kappa[i] (or None if stable).
    """
    sm_mass = QUARK_MASSES[quark]
    kappas = np.logspace(np.log10(kappa_min), np.log10(kappa_max), npoints)
    ees = np.full(npoints, np.nan)
    br_list = []

    for i, kappa in enumerate(kappas):
        mass = kappa * sm_mass
        brs = compute_quark_decay_brs(quark, mass)
        ees[i] = compute_quark_ee(brs)
        br_list.append(brs)

        if (i + 1) % 50 == 0 or i == 0:
            n_ch = len(brs) if brs else 0
            print(f"  [{quark}] {i + 1}/{npoints}: "
                  f"kappa={kappa:.4f}, m={mass:.4f} GeV, "
                  f"channels={n_ch}, EE={ees[i]:.4f}")

    return kappas, ees, br_list


def scan_all_quark_decays(
    quarks: list[str] | None = None,
    kappa_min: float = 0.01,
    kappa_max: float = 100.0,
    npoints: int = 200,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Scan all quarks' weak-decay EE.

    Returns:
        Dict mapping quark name -> (kappa_values, ee_values).
    """
    if quarks is None:
        quarks = ["u", "d", "s", "c", "b", "t"]

    results = {}
    for quark in quarks:
        print(f"\nScanning {quark} quark weak-decay EE...")
        kappas, ees, _ = scan_quark_decay(
            quark, kappa_min=kappa_min, kappa_max=kappa_max, npoints=npoints,
        )
        results[quark] = (kappas, ees)

    return results
