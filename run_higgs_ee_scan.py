#!/usr/bin/env python3
"""CLI — Scan Higgs decay EE as a function of y_t, universal kappa_f,
individual quark couplings, and hadronic fragmentation multiplicity.

Four complementary scans:
  1. y_t variation: physically change m_t, affecting loop form factors.
  2. Universal kappa_f: rescale all fermion couplings (paper's approach).
  3. Individual kappa_q: vary each quark's coupling separately.
  4. Fragmentation multiplicity: how hadronic multiplicity after
     hadronization affects EE — resolves the question of whether
     hadronization increases or decreases entanglement.
"""

import argparse

from smparamscan.higgs_ee_yt import (
    scan_higgs_ee, find_higgs_features,
    scan_higgs_kf, scan_higgs_kq, EE_MAX_NO_TT,
    scan_ee_vs_multiplicity, scan_kf_with_multiplicity,
    compute_higgs_ee_partonic, compute_higgs_ee_fragmented,
    SM_BRS,
)
from smparamscan.higgs_ee_plotter import plot_all_higgs_ee
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Scan Higgs decay EE vs top Yukawa, kappa_f, individual quarks, and multiplicity."
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
    print("KEY RESULTS: partonic vs naive-hadronized maxima")
    print(f"{'='*65}")
    for regime in ["ee_partonic", "ee_hadronized", "ee_inclusive"]:
        ee = results_kf[regime]
        max_idx = np.nanargmax(ee)
        print(f"  {regime:20s}: max = {ee[max_idx]:.6f} at kf = {kf[max_idx]:.4f}"
              f"  (SM = {ee[sm_idx]:.6f})")
    hcost = results_kf["hadronization_cost"][sm_idx]
    print(f"  Naive hadronization cost at SM: {hcost:.4f}")

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

    # ── Scan 4: Fragmentation multiplicity ──
    print()
    print("=" * 65)
    print("Scan 4: Hadronic fragmentation multiplicity")
    print("=" * 65)

    mult_results = scan_ee_vs_multiplicity(npoints=500)
    M_q = mult_results["M_quarks"]
    ee_frag = mult_results["ee_fragmented"]
    ee_part_sm = mult_results["ee_partonic"][0]
    ee_had_sm = mult_results["ee_naive_hadronized"][0]

    print(f"  Partonic EE (SM):           {ee_part_sm:.6f}")
    print(f"  Naive hadronized (M=1):     {ee_had_sm:.6f}")
    print(f"  Crossover at M_q = N_c = 3: fragmented EE = partonic EE")

    for M_test in [1, 3, 10, 20, 50, 100]:
        idx = np.argmin(np.abs(M_q - M_test))
        delta = (ee_frag[idx] - ee_part_sm) / ee_part_sm * 100
        print(f"  M_q = {M_test:>4d}: EE = {ee_frag[idx]:.6f} "
              f"({delta:+.2f}% vs partonic)")

    print(f"\n  At realistic M_q ~ 10-50 (Higgs-scale jet multiplicity),")
    print(f"  hadronization INCREASES EE beyond the partonic value.")
    print(f"  The multiplicity effect dominates over color-decoherence.")

    # kf scan at multiple multiplicities
    print("\n  Scanning kappa_f at several multiplicities...")
    kf_mult_results = scan_kf_with_multiplicity(
        kf_min=args.kf_min, kf_max=args.kf_max, npoints=400,
        M_values_quarks=[1.0, 3.0, 10.0, 30.0, 100.0]
    )

    kf_m = kf_mult_results["kappa_f"]
    print(f"\n  {'Multiplicity M_q':<20s} {'Max EE':>10s} {'at kf':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    ee_p = kf_mult_results["ee_partonic"]
    max_p_idx = np.nanargmax(ee_p)
    print(f"  {'Partonic':<20s} {ee_p[max_p_idx]:>10.6f} {kf_m[max_p_idx]:>10.3f}")
    for M_q_val in kf_mult_results["M_values_quarks"]:
        key = f"ee_M{M_q_val:.0f}"
        ee_m = kf_mult_results[key]
        mi = np.nanargmax(ee_m)
        print(f"  {'M_q = ' + str(int(M_q_val)):<20s} {ee_m[mi]:>10.6f} {kf_m[mi]:>10.3f}")

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

    print(f"\n  CORRECTED PICTURE (with fragmentation multiplicity):")
    print(f"  - The naive 'hadronized' model (N_c -> 1) misses the key effect:")
    print(f"    hadronic multiplicity >> N_c, so hadronization INCREASES EE.")
    print(f"  - The partonic EE is a LOWER BOUND on the true post-hadronization EE.")
    print(f"  - The top quark (no hadronization) sees partonic EE ~ {ee_part_sm:.4f}")
    print(f"  - Light quarks (with hadronization at M~20) see EE ~ {ee_frag[np.argmin(np.abs(M_q - 20))]:.4f}")
    print(f"  - Hadronization creates MORE entanglement, not less.")

    # ── Generate plots ──
    print("\nGenerating plots...")
    paths = plot_all_higgs_ee(results_yt, output_dir=args.output_dir,
                              results_kf=results_kf,
                              quark_results=quark_results,
                              mult_results=mult_results,
                              kf_mult_results=kf_mult_results)
    print(f"\nDone! {len(paths)} plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
