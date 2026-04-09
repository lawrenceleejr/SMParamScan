"""Plot entanglement entropy vs quark Yukawa coupling modifiers."""

import os

import matplotlib.pyplot as plt
import numpy as np

from .config import QUARK_LABELS, QUARK_NAMES


def plot_single_quark(
    quark: str,
    kappas: np.ndarray,
    ee_values: np.ndarray,
    output_dir: str = "plots",
    show: bool = False,
) -> str:
    """Plot EE vs kappa for a single quark.

    Returns:
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Filter out NaN values for clean line
    valid = ~np.isnan(ee_values)
    ax.plot(kappas[valid], ee_values[valid], "b-", linewidth=1.5)

    # Mark SM point (kappa = 1)
    sm_idx = np.argmin(np.abs(kappas - 1.0))
    ee_sm = ee_values[sm_idx]
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7, label=r"SM ($\kappa=1$)")
    ax.plot(1.0, ee_sm, "ro", markersize=8, zorder=5, label=f"SM: EE = {ee_sm:.4f}")

    # Mark EE maximum
    if valid.any():
        ee_max = np.nanmax(ee_values)
        kappa_max = kappas[np.nanargmax(ee_values)]
        ax.axhline(y=ee_max, color="green", linestyle=":", alpha=0.5,
                    label=f"Max EE = {ee_max:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel(QUARK_LABELS.get(quark, f"$\\kappa_{quark}$"), fontsize=14)
    ax.set_ylabel("Entanglement Entropy (EE)", fontsize=14)
    ax.set_title(f"EE vs {QUARK_NAMES.get(quark, quark)} quark Yukawa coupling",
                 fontsize=14)
    ax.legend(fontsize=11, loc="lower left")
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(output_dir, f"ee_vs_kappa_{quark}.pdf")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return filepath


def plot_all_quarks(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str = "plots",
    show: bool = False,
) -> list[str]:
    """Generate individual plots for each quark and a combined summary.

    Args:
        results: Dict from scan_all_quarks: quark -> (kappas, ee_values).
        output_dir: Directory to save plots.
        show: Whether to display plots interactively.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # Individual plots
    for quark, (kappas, ees) in results.items():
        path = plot_single_quark(quark, kappas, ees, output_dir, show=False)
        paths.append(path)
        print(f"  Saved: {path}")

    # Combined multi-panel figure
    quarks = list(results.keys())
    n = len(quarks)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                              squeeze=False)

    for idx, quark in enumerate(quarks):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        kappas, ees = results[quark]

        # Filter NaN for clean line
        valid = ~np.isnan(ees)
        ax.plot(kappas[valid], ees[valid], "b-", linewidth=1.5)

        # SM point
        sm_idx = np.argmin(np.abs(kappas - 1.0))
        ax.plot(1.0, ees[sm_idx], "ro", markersize=6, zorder=5)
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

        ax.set_xscale("log")
        ax.set_xlabel(QUARK_LABELS.get(quark, f"$\\kappa_{quark}$"), fontsize=12)
        ax.set_ylabel("EE", fontsize=12)
        ax.set_title(QUARK_NAMES.get(quark, quark).capitalize(), fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Entanglement Entropy vs Quark Yukawa Couplings",
                 fontsize=15, y=1.02)
    fig.tight_layout()

    combined_path = os.path.join(output_dir, "ee_vs_kappa_all_quarks.pdf")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    paths.append(combined_path)
    print(f"  Saved: {combined_path}")

    return paths
