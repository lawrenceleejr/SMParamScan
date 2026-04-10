#!/usr/bin/env python3
"""CLI — 2D scan of top quark and W boson decay EE over (m_W, m_t)."""

import argparse

from smparamscan.top_2d_scanner import scan_2d
from smparamscan.top_2d_plotter import plot_all_top_study


def main():
    parser = argparse.ArgumentParser(
        description="2D scan of top quark and W boson decay EE."
    )
    parser.add_argument("--mw-min", type=float, default=3.0,
                        help="Minimum W mass in GeV (default: 3.0)")
    parser.add_argument("--mw-max", type=float, default=300.0,
                        help="Maximum W mass in GeV (default: 300.0)")
    parser.add_argument("--n-mw", type=int, default=50,
                        help="Number of m_W scan points (default: 50)")
    parser.add_argument("--mt-min", type=float, default=3.0,
                        help="Minimum top mass in GeV (default: 3.0)")
    parser.add_argument("--mt-max", type=float, default=500.0,
                        help="Maximum top mass in GeV (default: 500.0)")
    parser.add_argument("--n-mt", type=int, default=50,
                        help="Number of m_t scan points (default: 50)")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory to save plots (default: plots/)")
    args = parser.parse_args()

    print(f"2D scan: m_W [{args.mw_min}, {args.mw_max}] ({args.n_mw} pts) x "
          f"m_t [{args.mt_min}, {args.mt_max}] ({args.n_mt} pts)")
    print(f"Total grid points: {args.n_mw * args.n_mt}")

    mw_grid, mt_grid, w_ee, top_ee = scan_2d(
        mw_min=args.mw_min, mw_max=args.mw_max, n_mw=args.n_mw,
        mt_min=args.mt_min, mt_max=args.mt_max, n_mt=args.n_mt,
    )

    print("\nGenerating plots...")
    paths = plot_all_top_study(mw_grid, mt_grid, w_ee, top_ee,
                               output_dir=args.output_dir)
    print(f"\nDone! {len(paths)} plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
