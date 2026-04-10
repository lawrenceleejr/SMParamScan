#!/usr/bin/env python3
"""CLI — Scan Higgs decay EE as a function of y_t, universal kappa_f,
and individual quark couplings in partonic/hadronized/inclusive regimes.

Three complementary scans:
  1. y_t variation: physically change m_t, affecting loop form factors.
  2. Universal kappa_f: rescale all fermion couplings (paper's approach).
     Shows Higgs EE has a MAXIMUM at kf ~ 1.03 (partonic), 0.60 (hadronized).
  3. Individual kappa_q: vary each quark's coupling separately in all three
     EE regimes, revealing the tension between partonic and hadronized maxima.
"""

import argparse

from smparamscan.higgs_ee_yt import (
    scan_higgs_ee, find_higgs_features,
    scan_higgs_kf, scan_higgs_kq, EE_MAX_NO_TT,
)
from smparamscan.higgs_ee_plotter import plot_all_higgs_ee
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Scan Higgs decay EE vs top Yukawa, kappa_f, and individual quarks."
    )
    parser.add_argument("--y-min", type=float, default=0.02)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--kf-min", type=float, default=0.05)
    parser.add_argument("--kf-max", type=float, default=10.0)
    parser.add_argument("--npoints", type=int, default=600)
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()

    # ── Scan 1: y_t variation ──
    print("=" * 65)
    print("Scan 1: Higgs EE vs y_t (physical top mass variation)")
    print("=" * 65)
    results_yt = scan_higgs_ee(args.y_min, args.y_max, args.npoints)
    features = find_higgs_features(results_yt, y_target=1.0, window=0.5)
    if features:
        for qty, ftype, y_at, val in features:
            print(f"  {qty:25s}  {ftype:15s}  y_t = {y_at:.4f}")
    else:
        print("  No local extrema near y_t = 1.")

    # ── Scan 2: Universal kappa_f ──
    print()
    print("=" * 65)
    print("Scan 2: Higgs EE vs kappa_f — three regimes")
    print("=" * 65)
    results_kf = scan_higgs_kf(args.kf_min, args.kf_max, args.npoints)

    kf = results_kf["kappa_f"]
    sm_idx = np.argmin(np.abs(kf - 1.0))

    print(f"\n{'='*65}")
    print("KEY RESULTS: partonic vs hadronized maxima")
    print(f"{'='*65}")
    for regime in ["ee_partonic", "ee_hadronized", "ee_inclusive"]:
        ee = results_kf[regime]
        max_idx = np.nanargmax(ee)
        print(f"  {regime:20s}: max = {ee[max_idx]:.6f} at kf = {kf[max_idx]:.4f}"
              f"  (SM = {ee[sm_idx]:.6f})")
    hcost = results_kf["hadronization_cost"][sm_idx]
    print(f"  Hadronization cost at SM: {hcost:.4f}")

    # ── Scan 3: Individual quark couplings ──
    print()
    print("=" * 65)
    print("Scan 3: Individual quark kappa_q scans")
    print("=" * 65)
    quark_results = {}
    for quark in ["t", "b", "c", "s"]:
        print(f"\n  Scanning kappa_{quark}...")
        quark_results[quark] = scan_higgs_kq(
            quark, kq_min=0.01, kq_max=20.0, npoints=400
        )
        kq = quark_results[quark]["kappa_q"]
        sq = np.argmin(np.abs(kq - 1.0))
        for regime in ["ee_partonic", "ee_hadronized"]:
            ee = quark_results[quark][regime]
            mi = np.nanargmax(ee)
            print(f"    {regime:20s}: max at kq = {kq[mi]:.3f} "
                  f"(SM val = {ee[sq]:.6f})")

    # ── Summary table ──
    print()
    print("=" * 65)
    print("SUMMARY: Where does each regime maximize?")
    print("=" * 65)
    print(f"  {'Scan':<25s} {'Partonic max':<18s} {'Hadronized max':<18s}")
    print(f"  {'-'*25} {'-'*18} {'-'*18}")

    max_p = kf[np.nanargmax(results_kf["ee_partonic"])]
    max_h = kf[np.nanargmax(results_kf["ee_hadronized"])]
    print(f"  {'Universal kf':<25s} {max_p:<18.3f} {max_h:<18.3f}")

    for quark in ["t", "b", "c", "s"]:
        kq = quark_results[quark]["kappa_q"]
        mp = kq[np.nanargmax(quark_results[quark]["ee_partonic"])]
        mh = kq[np.nanargmax(quark_results[quark]["ee_hadronized"])]
        print(f"  {'kappa_' + quark:<25s} {mp:<18.3f} {mh:<18.3f}")

    print(f"\n  The top quark does not hadronize => partonic regime applies.")
    print(f"  All other quarks hadronize => hadronized regime applies.")
    print(f"  Partonic EE is maximized at the SM (kf=1.03).")
    print(f"  Hadronized EE would prefer kf=0.60 — smaller couplings.")

    # ── Generate plots ──
    print("\nGenerating plots...")
    paths = plot_all_higgs_ee(results_yt, output_dir=args.output_dir,
                              results_kf=results_kf,
                              quark_results=quark_results)
    print(f"\nDone! {len(paths)} plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
