"""Parameter scan orchestration — vary quark Yukawas and compute EE."""

import shutil

import numpy as np

from .config import SM_PARAMS, QUARK_MASS_PARAMS
from .docker_manager import build_image, run_hdecay
from .hdecay_input import generate_input
from .hdecay_output import parse_output
from .entropy import compute_ee, compute_ee_hadronized, compute_ee_inclusive


# Keys for the three EE definitions
EE_PARTONIC = "partonic"
EE_HADRONIZED = "hadronized"
EE_INCLUSIVE = "inclusive"

EE_FUNCTIONS = {
    EE_PARTONIC: compute_ee,
    EE_HADRONIZED: compute_ee_hadronized,
    EE_INCLUSIVE: compute_ee_inclusive,
}


def _compute_all_ees(brs: dict[str, float]) -> dict[str, float]:
    """Compute all three EE definitions from a single set of BRs."""
    return {key: fn(brs) for key, fn in EE_FUNCTIONS.items()}


def scan_quark(
    quark: str,
    kappa_min: float = 0.01,
    kappa_max: float = 100.0,
    npoints: int = 100,
    no_rebuild: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Scan a single quark Yukawa coupling and compute EE at each point.

    Returns:
        Tuple of (kappa_values, ee_dict) where ee_dict maps
        EE definition name -> array of EE values.
    """
    mass_param = QUARK_MASS_PARAMS.get(quark)
    kappas = np.logspace(np.log10(kappa_min), np.log10(kappa_max), npoints)

    if mass_param is None:
        print(f"Warning: quark '{quark}' has no mass parameter in HDECAY. "
              f"u and d quarks are treated as massless — EE will be constant.")
        sm_input = generate_input()
        work_dir = run_hdecay(sm_input)
        brs = parse_output(work_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        sm_ees = _compute_all_ees(brs)
        return kappas, {k: np.full_like(kappas, v) for k, v in sm_ees.items()}

    sm_mass = SM_PARAMS[mass_param]
    ee_arrays = {k: np.zeros(npoints) for k in EE_FUNCTIONS}

    if not no_rebuild:
        build_image()

    for i, kappa in enumerate(kappas):
        mass = kappa * sm_mass
        input_content = generate_input({mass_param: mass})

        try:
            work_dir = run_hdecay(input_content)
            brs = parse_output(work_dir)
            ees = _compute_all_ees(brs)
            for k in ee_arrays:
                ee_arrays[k][i] = ees[k]
            shutil.rmtree(work_dir, ignore_errors=True)
        except RuntimeError as e:
            print(f"  Warning: HDECAY failed at kappa={kappa:.4f}: {e}")
            for k in ee_arrays:
                ee_arrays[k][i] = np.nan

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{quark}] {i + 1}/{npoints}: "
                  f"kappa={kappa:.4f}, "
                  f"EE_part={ee_arrays[EE_PARTONIC][i]:.6f}, "
                  f"EE_had={ee_arrays[EE_HADRONIZED][i]:.6f}, "
                  f"EE_incl={ee_arrays[EE_INCLUSIVE][i]:.6f}")

    return kappas, ee_arrays


def scan_all_quarks(
    quarks: list[str] | None = None,
    kappa_min: float = 0.01,
    kappa_max: float = 100.0,
    npoints: int = 100,
) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Scan all specified quarks.

    Returns:
        Dict mapping quark name -> (kappa_values, ee_dict).
    """
    if quarks is None:
        quarks = ["u", "d", "s", "c", "b", "t"]

    build_image()

    results = {}
    for quark in quarks:
        print(f"\nScanning {quark} quark Yukawa coupling...")
        kappas, ee_dict = scan_quark(
            quark,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            npoints=npoints,
            no_rebuild=True,
        )
        results[quark] = (kappas, ee_dict)

    return results
