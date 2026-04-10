"""Plot entanglement entropy from quark weak decays vs Yukawa coupling."""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .config import QUARK_LABELS, QUARK_NAMES
from .quark_decay import QUARK_MASSES

# Higgs VEV in GeV
V_HIGGS = 246.22  # GeV


def _add_secondary_axes(ax, quark):
    """Add top axes showing Yukawa coupling y_q and mass m_q (GeV)."""
    m_sm = QUARK_MASSES[quark]

    # Top axis 1: mass in GeV  (m = kappa * m_SM)
    ax_mass = ax.secondary_xaxis(
        "top", functions=(lambda k: k * m_sm, lambda m: m / m_sm)
    )
    ax_mass.set_xlabel(f"$m_{{{quark}}}$ [GeV]", fontsize=12, labelpad=6)
    ax_mass.tick_params(labelsize=10)

    # Top axis 2: Yukawa y_q = sqrt(2) * m / v = sqrt(2) * kappa * m_SM / v
    y_sm = np.sqrt(2) * m_sm / V_HIGGS

    ax_yuk = ax.secondary_xaxis(
        1.18, functions=(lambda k: k * y_sm, lambda y: y / y_sm)
    )
    ax_yuk.set_xlabel(f"$y_{{{quark}}}$", fontsize=12, labelpad=6)
    ax_yuk.tick_params(labelsize=10)

    return ax_mass, ax_yuk


def plot_single_quark_decay(
    quark: str,
    kappas: np.ndarray,
    ees: np.ndarray,
    output_dir: str = "plots",
    show: bool = False,
) -> str:
    """Plot quark-decay EE vs kappa for a single quark."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    valid = ~np.isnan(ees)
    if valid.any():
        ax.plot(kappas[valid], ees[valid], color="#1f77b4",
                linewidth=2, label="Quark decay EE")

        # SM point
        sm_idx = np.argmin(np.abs(kappas - 1.0))
        if not np.isnan(ees[sm_idx]):
            ax.plot(1.0, ees[sm_idx], "o", color="#1f77b4",
                    markersize=8, zorder=5, label=f"SM (EE={ees[sm_idx]:.3f})")

        # Maximum
        ee_max = np.nanmax(ees)
        kappa_max = kappas[np.nanargmax(ees)]
        ax.plot(kappa_max, ee_max, "*", color="#d62728",
                markersize=12, zorder=6, markeredgecolor="k",
                markeredgewidth=0.5,
                label=f"Max EE={ee_max:.3f} at $\\kappa$={kappa_max:.2f}")

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel(QUARK_LABELS.get(quark, f"$\\kappa_{quark}$"), fontsize=14)
    ax.set_ylabel("Entanglement Entropy (EE)", fontsize=14)
    ax.set_title(
        f"Weak-Decay EE vs {QUARK_NAMES.get(quark, quark)} quark Yukawa coupling",
        fontsize=14, pad=60,
    )
    ax.legend(fontsize=10, loc="best")
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    _add_secondary_axes(ax, quark)

    fig.tight_layout()
    filepath = os.path.join(output_dir, f"quark_decay_ee_{quark}.pdf")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return filepath


def plot_all_quark_decays(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str = "plots",
    show: bool = False,
) -> list[str]:
    """Generate individual and combined quark-decay EE plots."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # Individual plots
    for quark, (kappas, ees) in results.items():
        path = plot_single_quark_decay(quark, kappas, ees, output_dir)
        paths.append(path)
        print(f"  Saved: {path}")

    # Combined multi-panel figure
    quarks = list(results.keys())
    n = len(quarks)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows),
                              squeeze=False)

    for idx, quark in enumerate(quarks):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        kappas, ees = results[quark]

        valid = ~np.isnan(ees)
        if valid.any():
            ax.plot(kappas[valid], ees[valid], color="#1f77b4",
                    linewidth=1.5)

            # SM point
            sm_idx = np.argmin(np.abs(kappas - 1.0))
            if not np.isnan(ees[sm_idx]):
                ax.plot(1.0, ees[sm_idx], "o", color="#1f77b4",
                        markersize=5, zorder=5)

            # Max marker
            ax.plot(kappas[np.nanargmax(ees)], np.nanmax(ees), "*",
                    color="#d62728", markersize=8, zorder=6,
                    markeredgecolor="k", markeredgewidth=0.3)

        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlabel(QUARK_LABELS.get(quark, f"$\\kappa_{quark}$"),
                       fontsize=12)
        ax.set_ylabel("EE", fontsize=12)
        ax.set_title(QUARK_NAMES.get(quark, quark).capitalize(),
                      fontsize=13, pad=50)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        _add_secondary_axes(ax, quark)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Quark Weak-Decay Entanglement Entropy vs Yukawa Couplings",
                 fontsize=15, y=1.02)
    fig.tight_layout(h_pad=4.0)

    combined_path = os.path.join(output_dir, "quark_decay_ee_all.pdf")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    paths.append(combined_path)
    print(f"  Saved: {combined_path}")

    return paths
