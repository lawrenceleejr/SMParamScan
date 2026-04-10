"""Plot entanglement entropy vs quark Yukawa coupling modifiers."""

import os

import matplotlib.pyplot as plt
import numpy as np

from .config import QUARK_LABELS, QUARK_NAMES
from .scanner import EE_PARTONIC, EE_HADRONIZED, EE_INCLUSIVE

# Plot styling for each EE definition
EE_STYLES = {
    EE_PARTONIC:   {"color": "#1f77b4", "ls": "-",  "label": "Partonic"},
    EE_HADRONIZED: {"color": "#d62728", "ls": "--", "label": "Hadronized ($N_c$=1)"},
    EE_INCLUSIVE:  {"color": "#2ca02c", "ls": "-.", "label": "Inclusive hadronic"},
}


def plot_single_quark(
    quark: str,
    kappas: np.ndarray,
    ee_dict: dict[str, np.ndarray],
    output_dir: str = "plots",
    show: bool = False,
) -> str:
    """Plot EE vs kappa for a single quark with all three definitions."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for key, ees in ee_dict.items():
        style = EE_STYLES[key]
        valid = ~np.isnan(ees)
        ax.plot(kappas[valid], ees[valid], color=style["color"],
                ls=style["ls"], linewidth=1.8, label=style["label"])

        # Mark SM point
        sm_idx = np.argmin(np.abs(kappas - 1.0))
        if not np.isnan(ees[sm_idx]):
            ax.plot(1.0, ees[sm_idx], "o", color=style["color"],
                    markersize=7, zorder=5)

    # Mark maximum of each curve
    for key, ees in ee_dict.items():
        style = EE_STYLES[key]
        valid = ~np.isnan(ees)
        if valid.any():
            ee_max = np.nanmax(ees)
            kappa_at_max = kappas[np.nanargmax(ees)]
            ax.plot(kappa_at_max, ee_max, "*", color=style["color"],
                    markersize=10, zorder=6, markeredgecolor="k",
                    markeredgewidth=0.5)

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5,
               label=r"SM ($\kappa=1$)")

    ax.set_xscale("log")
    ax.set_xlabel(QUARK_LABELS.get(quark, f"$\\kappa_{quark}$"), fontsize=14)
    ax.set_ylabel("Entanglement Entropy (EE)", fontsize=14)
    ax.set_title(f"EE vs {QUARK_NAMES.get(quark, quark)} quark Yukawa coupling",
                 fontsize=14)
    ax.legend(fontsize=10, loc="best")
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
    results: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    output_dir: str = "plots",
    show: bool = False,
) -> list[str]:
    """Generate individual plots and a combined summary."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # Individual plots
    for quark, (kappas, ee_dict) in results.items():
        path = plot_single_quark(quark, kappas, ee_dict, output_dir)
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
        kappas, ee_dict = results[quark]

        for key, ees in ee_dict.items():
            style = EE_STYLES[key]
            valid = ~np.isnan(ees)
            ax.plot(kappas[valid], ees[valid], color=style["color"],
                    ls=style["ls"], linewidth=1.5,
                    label=style["label"] if idx == 0 else None)

            # SM point
            sm_idx = np.argmin(np.abs(kappas - 1.0))
            if not np.isnan(ees[sm_idx]):
                ax.plot(1.0, ees[sm_idx], "o", color=style["color"],
                        markersize=5, zorder=5)

            # Max marker
            if valid.any():
                ax.plot(kappas[np.nanargmax(ees)], np.nanmax(ees), "*",
                        color=style["color"], markersize=8, zorder=6,
                        markeredgecolor="k", markeredgewidth=0.3)

        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.4)
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

    # Shared legend from first subplot
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.04))

    fig.suptitle("Entanglement Entropy vs Quark Yukawa Couplings",
                 fontsize=15, y=1.07)
    fig.tight_layout()

    combined_path = os.path.join(output_dir, "ee_vs_kappa_all_quarks.pdf")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    paths.append(combined_path)
    print(f"  Saved: {combined_path}")

    return paths
