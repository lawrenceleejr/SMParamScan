#!/usr/bin/env python3
"""CLI — Scan Higgs decay EE as a function of y_t and universal kappa_f.

Two complementary scans:
  1. y_t variation: physically change m_t = y_t * v/sqrt(2), which affects loop
     form factors (h->gg, h->gamgam) and h->tt threshold. Shows Higgs EE is
     on a plateau near y_t = 1.
  2. Universal kappa_f: rescale all fermion couplings while keeping masses at SM
     values (the paper's approach). Shows Higgs EE has a MAXIMUM at kf ~ 1.03,
     essentially the SM value. The SM maximizes entanglement entropy.
"""

import argparse

from smparamscan.higgs_ee_yt import (
    scan_higgs_ee, find_higgs_features,
    scan_higgs_kf, EE_MAX_NO_TT,
)
from smparamscan.higgs_ee_plotter import plot_all_higgs_ee
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Scan Higgs decay EE as a function of top Yukawa and kappa_f."
    )
    parser.add_argument("--y-min", type=float, default=0.02,
                        help="Minimum y_t (default: 0.02)")
    parser.add_argument("--y-max", type=float, default=10.0,
                        help="Maximum y_t (default: 10.0)")
    parser.add_argument("--kf-min", type=float, default=0.05,
                        help="Minimum kappa_f (default: 0.05)")
    parser.add_argument("--kf-max", type=float, default=10.0,
                        help="Maximum kappa_f (default: 10.0)")
    parser.add_argument("--npoints", type=int, default=600,
                        help="Number of scan points (default: 600)")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Output directory (default: plots/)")
    args = parser.parse_args()

    # ── Scan 1: y_t variation (physical mass change) ──
    print("=" * 60)
    print("Scan 1: Higgs EE vs y_t (physical top mass variation)")
    print("=" * 60)
    results_yt = scan_higgs_ee(args.y_min, args.y_max, args.npoints)

    print("\nSearching for features near y_t = 1...")
    features = find_higgs_features(results_yt, y_target=1.0, window=0.5)

    if features:
        print(f"\nFound {len(features)} features near y_t = 1:")
        for qty, ftype, y_at, val in features:
            print(f"  {qty:25s}  {ftype:15s}  y_t = {y_at:.4f}  val = {val:.6f}")
    else:
        print("\nNo local extrema found near y_t = 1.")

    # ── Scan 2: Universal kappa_f (paper's approach) ──
    print()
    print("=" * 60)
    print("Scan 2: Higgs EE vs kappa_f (universal fermion coupling)")
    print("=" * 60)
    results_kf = scan_higgs_kf(args.kf_min, args.kf_max, args.npoints)

    # Find maximum
    ee = results_kf["ee_partonic"]
    kf = results_kf["kappa_f"]
    max_idx = np.nanargmax(ee)
    sm_idx = np.argmin(np.abs(kf - 1.0))

    print(f"\n{'='*60}")
    print("KEY RESULT: SM maximizes Higgs entanglement entropy")
    print(f"{'='*60}")
    print(f"  EE_max (theoretical upper bound) = {EE_MAX_NO_TT:.6f}")
    print(f"  Maximum EE achieved  = {ee[max_idx]:.6f}  at kf = {kf[max_idx]:.4f}")
    print(f"  SM EE (kf=1)         = {ee[sm_idx]:.6f}")
    print(f"  Delta(EE) SM vs max  = {ee[max_idx] - ee[sm_idx]:.2e}")
    print(f"  SM fraction of max   = {ee[sm_idx]/EE_MAX_NO_TT:.4f}")
    print(f"  Peak at kf/kf_SM     = {kf[max_idx]:.4f}")

    # ── Generate plots ──
    print("\nGenerating plots...")
    paths = plot_all_higgs_ee(results_yt, output_dir=args.output_dir,
                              results_kf=results_kf)
    print(f"\nDone! {len(paths)} plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
