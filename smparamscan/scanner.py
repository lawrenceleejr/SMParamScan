"""Parameter scan orchestration — vary quark Yukawas and compute EE."""

import shutil

import numpy as np

from .config import SM_PARAMS, QUARK_MASS_PARAMS
from .docker_manager import build_image, run_hdecay
from .hdecay_input import generate_input
from .hdecay_output import parse_output
from .entropy import compute_ee


def scan_quark(
    quark: str,
    kappa_min: float = 0.01,
    kappa_max: float = 100.0,
    npoints: int = 100,
    no_rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Scan a single quark Yukawa coupling and compute EE at each point.

    The Yukawa coupling is varied by scaling the quark mass:
        m_q -> kappa * m_q^SM

    Args:
        quark: Quark name ('u', 'd', 's', 'c', 'b', 't').
        kappa_min: Minimum kappa value.
        kappa_max: Maximum kappa value.
        npoints: Number of scan points (log-spaced).
        no_rebuild: Skip Docker rebuild check.

    Returns:
        Tuple of (kappa_values, ee_values) arrays.
    """
    mass_param = QUARK_MASS_PARAMS.get(quark)
    if mass_param is None:
        print(f"Warning: quark '{quark}' has no mass parameter in HDECAY. "
              f"u and d quarks are treated as massless — EE will be constant.")
        # Return SM EE as a flat line
        kappas = np.logspace(np.log10(kappa_min), np.log10(kappa_max), npoints)
        sm_input = generate_input()
        work_dir = run_hdecay(sm_input)
        brs = parse_output(work_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        ee_sm = compute_ee(brs)
        return kappas, np.full_like(kappas, ee_sm)

    sm_mass = SM_PARAMS[mass_param]
    kappas = np.logspace(np.log10(kappa_min), np.log10(kappa_max), npoints)
    ee_values = np.zeros(npoints)

    if not no_rebuild:
        build_image()

    for i, kappa in enumerate(kappas):
        mass = kappa * sm_mass
        overrides = {mass_param: mass}
        input_content = generate_input(overrides)

        try:
            work_dir = run_hdecay(input_content)
            brs = parse_output(work_dir)
            ee_values[i] = compute_ee(brs)
            shutil.rmtree(work_dir, ignore_errors=True)
        except RuntimeError as e:
            print(f"  Warning: HDECAY failed at kappa={kappa:.4f}: {e}")
            ee_values[i] = np.nan

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{quark}] {i + 1}/{npoints}: "
                  f"kappa={kappa:.4f}, EE={ee_values[i]:.6f}")

    return kappas, ee_values


def scan_all_quarks(
    quarks: list[str] | None = None,
    kappa_min: float = 0.01,
    kappa_max: float = 100.0,
    npoints: int = 100,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Scan all specified quarks.

    Args:
        quarks: List of quark names. Defaults to all 6.
        kappa_min: Minimum kappa value.
        kappa_max: Maximum kappa value.
        npoints: Number of scan points per quark.

    Returns:
        Dict mapping quark name -> (kappa_values, ee_values).
    """
    if quarks is None:
        quarks = ["u", "d", "s", "c", "b", "t"]

    build_image()

    results = {}
    for quark in quarks:
        print(f"\nScanning {quark} quark Yukawa coupling...")
        kappas, ees = scan_quark(
            quark,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            npoints=npoints,
            no_rebuild=True,
        )
        results[quark] = (kappas, ees)

    return results
