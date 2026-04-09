#!/usr/bin/env python3
"""CLI entry point for SMParamScan — scan quark Yukawa couplings and plot EE."""

import argparse

from smparamscan.scanner import scan_all_quarks
from smparamscan.plotter import plot_all_quarks


def main():
    parser = argparse.ArgumentParser(
        description="Scan SM quark Yukawa couplings and plot entanglement entropy."
    )
    parser.add_argument(
        "--quarks", type=str, default="u,d,s,c,b,t",
        help="Comma-separated list of quarks to scan (default: u,d,s,c,b,t)",
    )
    parser.add_argument(
        "--npoints", type=int, default=100,
        help="Number of scan points per quark (default: 100)",
    )
    parser.add_argument(
        "--kappa-min", type=float, default=0.01,
        help="Minimum kappa value (default: 0.01)",
    )
    parser.add_argument(
        "--kappa-max", type=float, default=100.0,
        help="Maximum kappa value (default: 100.0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots",
        help="Directory to save plots (default: plots/)",
    )
    args = parser.parse_args()

    quarks = [q.strip() for q in args.quarks.split(",")]
    print(f"Scanning quarks: {quarks}")
    print(f"kappa range: [{args.kappa_min}, {args.kappa_max}], {args.npoints} points")

    results = scan_all_quarks(
        quarks=quarks,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        npoints=args.npoints,
    )

    print("\nGenerating plots...")
    paths = plot_all_quarks(results, output_dir=args.output_dir)

    print(f"\nDone! {len(paths)} plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
