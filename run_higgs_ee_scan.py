#!/usr/bin/env python3
"""CLI — Scan Higgs decay EE as a function of y_t, universal kappa_f,
individual quark couplings, hadronic fragmentation multiplicity, and M_W.

Five complementary scans:
  1. y_t variation: physically change m_t, affecting loop form factors.
  2. Universal kappa_f: rescale all fermion couplings (paper's approach).
  3. Individual kappa_q: vary each quark's coupling separately.
  4. Fragmentation multiplicity: how hadronic multiplicity after
     hadronization affects EE.
  5. M_W variation: how the W boson mass affects Higgs EE through
     the WW* channel, reproducing the paper's W mass result.
"""

import argparse

from smparamscan.higgs_ee_yt import (
    scan_higgs_ee, find_higgs_features,
    scan_higgs_kf, scan_higgs_kq, EE_MAX_NO_TT,
    scan_ee_vs_multiplicity, scan_kf_with_multiplicity,
    scan_kq_with_multiplicity, scan_higgs_ee_mw,
    SM_BRS,
)
from smparamscan.higgs_ee_plotter import plot_all_higgs_ee
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Scan Higgs decay EE vs top Yukawa, kappa_f, individual quarks, multiplicity, and M_W."
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

    # kappa_t and kappa_b at multiple multiplicities
    print("\n  Scanning kappa_t and kappa_b at several multiplicities...")
    kq_mult_results = {}
    for quark in ["t", "b"]:
        kq_mult_results[quark] = scan_kq_with_multiplicity(
            quark, kq_min=0.01, kq_max=20.0, npoints=400,
            M_values_quarks=[1.0, 3.0, 10.0, 30.0, 100.0]
        )
        kq_arr = kq_mult_results[quark]["kappa_q"]
        print(f"\n  kappa_{quark} multiplicity scan:")
        ee_p_q = kq_mult_results[quark]["ee_partonic"]
        mp = np.nanargmax(ee_p_q)
        print(f"    {'Partonic':<20s} max EE={ee_p_q[mp]:.6f} at k{quark}={kq_arr[mp]:.3f}")
        for M_q_val in [1.0, 3.0, 10.0, 30.0, 100.0]:
            key = f"ee_M{M_q_val:.0f}"
            ee_m = kq_mult_results[quark][key]
            mi = np.nanargmax(ee_m)
            print(f"    {'M_q = ' + str(int(M_q_val)):<20s} max EE={ee_m[mi]:.6f} "
                  f"at k{quark}={kq_arr[mi]:.3f}")

    # ── Scan 5: M_W variation ──
    print()
    print("=" * 65)
    print("Scan 5: Higgs EE vs M_W")
    print("=" * 65)
    mw_results = scan_higgs_ee_mw(mw_min=20.0, mw_max=120.0, npoints=400)

    mw = mw_results["m_W"]
    ee_mw = mw_results["ee_partonic"]
    ee_mw_frag = mw_results["ee_fragmented"]
    sm_mw_idx = np.argmin(np.abs(mw - 80.379))
    max_mw_idx = np.nanargmax(ee_mw)

    print(f"  SM M_W = 80.379 GeV: partonic EE = {ee_mw[sm_mw_idx]:.6f}")
    print(f"  Partonic max: EE = {ee_mw[max_mw_idx]:.6f} at M_W = {mw[max_mw_idx]:.1f} GeV")

    max_mw_f = np.nanargmax(ee_mw_frag)
    print(f"  Fragmented max (M_q=20): EE = {ee_mw_frag[max_mw_f]:.6f} "
          f"at M_W = {mw[max_mw_f]:.1f} GeV")

    # BR at SM
    print(f"\n  BRs at SM M_W:")
    for ch in ["WW", "bb", "gg", "ZZ"]:
        br = mw_results[f"br_{ch}"][sm_mw_idx]
        print(f"    BR({ch}) = {br:.4f}")

    # ── Generate plots ──
    print("\nGenerating plots...")
    paths = plot_all_higgs_ee(results_yt, output_dir=args.output_dir,
                              results_kf=results_kf,
                              quark_results=quark_results,
                              mult_results=mult_results,
                              kf_mult_results=kf_mult_results,
                              kq_mult_results=kq_mult_results,
                              mw_results=mw_results)
    print(f"\nDone! {len(paths)} plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
