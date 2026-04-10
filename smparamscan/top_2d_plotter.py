"""Plot 2D heatmaps of W and top decay EE over (m_W, m_t) plane."""

import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .quark_decay import QUARK_MASSES

SM_MW = 80.379
SM_MT = 172.5
M_B = QUARK_MASSES["b"]


def plot_w_ee_2d(
    mw_grid: np.ndarray,
    mt_grid: np.ndarray,
    w_ee: np.ndarray,
    output_dir: str = "plots",
    show: bool = False,
) -> str:
    """2D heatmap of W decay EE in the (m_W, m_t) plane."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    MW, MT = np.meshgrid(mw_grid, mt_grid)
    im = ax.pcolormesh(MW, MT, w_ee, cmap="viridis", shading="gouraud")
    cb = fig.colorbar(im, ax=ax, label="W Decay EE", pad=0.02)

    # Contour lines
    valid = ~np.isnan(w_ee)
    if valid.any():
        levels = np.linspace(np.nanmin(w_ee), np.nanmax(w_ee), 12)
        cs = ax.contour(MW, MT, w_ee, levels=levels, colors="w",
                        linewidths=0.6, alpha=0.6)
        ax.clabel(cs, fontsize=7, fmt="%.2f")

    # tb threshold: m_W = m_t + m_b (above this line, W -> tb opens)
    mt_line = np.linspace(mt_grid[0], mt_grid[-1], 200)
    mw_tb = mt_line + M_B
    mask = (mw_tb >= mw_grid[0]) & (mw_tb <= mw_grid[-1])
    ax.plot(mw_tb[mask], mt_line[mask], "--", color="red", linewidth=1.5,
            label=r"$m_W = m_t + m_b$ (tb threshold)")

    # SM point
    ax.plot(SM_MW, SM_MT, "*", color="red", markersize=14, zorder=10,
            markeredgecolor="k", markeredgewidth=0.8, label="SM")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_W$ [GeV]", fontsize=14)
    ax.set_ylabel(r"$m_t$ [GeV]", fontsize=14)
    ax.set_title("W Boson Decay Entanglement Entropy", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    path = os.path.join(output_dir, "w_decay_ee_2d.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_top_ee_2d(
    mw_grid: np.ndarray,
    mt_grid: np.ndarray,
    top_ee: np.ndarray,
    output_dir: str = "plots",
    show: bool = False,
) -> str:
    """2D heatmap of top decay EE in the (m_W, m_t) plane."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    MW, MT = np.meshgrid(mw_grid, mt_grid)
    im = ax.pcolormesh(MW, MT, top_ee, cmap="viridis", shading="gouraud")
    cb = fig.colorbar(im, ax=ax, label="Top Decay EE", pad=0.02)

    # Contour lines
    valid = ~np.isnan(top_ee)
    if valid.any():
        levels = np.linspace(np.nanmin(top_ee[valid]), np.nanmax(top_ee[valid]), 12)
        cs = ax.contour(MW, MT, top_ee, levels=levels, colors="w",
                        linewidths=0.6, alpha=0.6)
        ax.clabel(cs, fontsize=7, fmt="%.3f")

    # On-shell W threshold for top: m_t = m_W + m_b
    mw_line = np.linspace(mw_grid[0], mw_grid[-1], 200)
    mt_onshell = mw_line + M_B
    mask = (mt_onshell >= mt_grid[0]) & (mt_onshell <= mt_grid[-1])
    ax.plot(mw_line[mask], mt_onshell[mask], "--", color="red", linewidth=1.5,
            label=r"$m_t = m_W + m_b$ (on-shell $W$)")

    # SM point
    ax.plot(SM_MW, SM_MT, "*", color="red", markersize=14, zorder=10,
            markeredgecolor="k", markeredgewidth=0.8, label="SM")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_W$ [GeV]", fontsize=14)
    ax.set_ylabel(r"$m_t$ [GeV]", fontsize=14)
    ax.set_title("Top Quark Decay Entanglement Entropy", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    path = os.path.join(output_dir, "top_decay_ee_2d.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_w_ee_1d(
    mw_grid: np.ndarray,
    mt_grid: np.ndarray,
    w_ee: np.ndarray,
    output_dir: str = "plots",
    show: bool = False,
) -> str:
    """1D slice: W decay EE vs m_W at SM m_t."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Find the m_t slice closest to SM
    sm_mt_idx = np.argmin(np.abs(mt_grid - SM_MT))
    ees = w_ee[sm_mt_idx, :]
    valid = ~np.isnan(ees)

    if valid.any():
        ax.plot(mw_grid[valid], ees[valid], color="#1f77b4", linewidth=2,
                label=f"W decay EE ($m_t$={mt_grid[sm_mt_idx]:.0f} GeV)")

        # SM point
        sm_mw_idx = np.argmin(np.abs(mw_grid - SM_MW))
        if not np.isnan(ees[sm_mw_idx]):
            ax.plot(SM_MW, ees[sm_mw_idx], "o", color="#1f77b4",
                    markersize=8, zorder=5,
                    label=f"SM (EE={ees[sm_mw_idx]:.3f})")

        # Maximum
        ee_max = np.nanmax(ees)
        mw_at_max = mw_grid[np.nanargmax(ees)]
        ax.plot(mw_at_max, ee_max, "*", color="#d62728",
                markersize=12, zorder=6, markeredgecolor="k",
                markeredgewidth=0.5,
                label=f"Max EE={ee_max:.3f} at $m_W$={mw_at_max:.1f} GeV")

    ax.axvline(x=SM_MW, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$m_W$ [GeV]", fontsize=14)
    ax.set_ylabel("Entanglement Entropy (EE)", fontsize=14)
    ax.set_title("W Boson Decay EE vs $m_W$", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "w_decay_ee_vs_mw.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_combined(
    mw_grid: np.ndarray,
    mt_grid: np.ndarray,
    w_ee: np.ndarray,
    top_ee: np.ndarray,
    output_dir: str = "plots",
    show: bool = False,
) -> str:
    """Combined 3-panel figure: W EE 1D, W EE 2D, Top EE 2D."""
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(18, 6))

    MW, MT = np.meshgrid(mw_grid, mt_grid)

    # Panel 1: W EE vs m_W (1D slice at SM m_t)
    ax1 = fig.add_subplot(131)
    sm_mt_idx = np.argmin(np.abs(mt_grid - SM_MT))
    ees = w_ee[sm_mt_idx, :]
    valid = ~np.isnan(ees)
    if valid.any():
        ax1.plot(mw_grid[valid], ees[valid], color="#1f77b4", linewidth=2)
        sm_mw_idx = np.argmin(np.abs(mw_grid - SM_MW))
        if not np.isnan(ees[sm_mw_idx]):
            ax1.plot(SM_MW, ees[sm_mw_idx], "*", color="red", markersize=12,
                     zorder=6, markeredgecolor="k", markeredgewidth=0.5)
    ax1.axvline(x=SM_MW, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$m_W$ [GeV]", fontsize=12)
    ax1.set_ylabel("EE", fontsize=12)
    ax1.set_title(r"W Decay EE vs $m_W$", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)

    # Panel 2: W EE 2D heatmap
    ax2 = fig.add_subplot(132)
    im2 = ax2.pcolormesh(MW, MT, w_ee, cmap="viridis", shading="gouraud")
    fig.colorbar(im2, ax=ax2, label="W EE", pad=0.02, fraction=0.046)
    mt_line = np.linspace(mt_grid[0], mt_grid[-1], 200)
    mw_tb = mt_line + M_B
    mask = (mw_tb >= mw_grid[0]) & (mw_tb <= mw_grid[-1])
    ax2.plot(mw_tb[mask], mt_line[mask], "--", color="red", linewidth=1.2,
             alpha=0.8)
    ax2.plot(SM_MW, SM_MT, "*", color="red", markersize=10, zorder=10,
             markeredgecolor="k", markeredgewidth=0.6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$m_W$ [GeV]", fontsize=12)
    ax2.set_ylabel(r"$m_t$ [GeV]", fontsize=12)
    ax2.set_title("W Decay EE", fontsize=13)
    ax2.tick_params(labelsize=10)

    # Panel 3: Top EE 2D heatmap
    ax3 = fig.add_subplot(133)
    im3 = ax3.pcolormesh(MW, MT, top_ee, cmap="viridis", shading="gouraud")
    fig.colorbar(im3, ax=ax3, label="Top EE", pad=0.02, fraction=0.046)
    mw_line = np.linspace(mw_grid[0], mw_grid[-1], 200)
    mt_onshell = mw_line + M_B
    mask = (mt_onshell >= mt_grid[0]) & (mt_onshell <= mt_grid[-1])
    ax3.plot(mw_line[mask], mt_onshell[mask], "--", color="red",
             linewidth=1.2, alpha=0.8)
    ax3.plot(SM_MW, SM_MT, "*", color="red", markersize=10, zorder=10,
             markeredgecolor="k", markeredgewidth=0.6)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$m_W$ [GeV]", fontsize=12)
    ax3.set_ylabel(r"$m_t$ [GeV]", fontsize=12)
    ax3.set_title("Top Decay EE", fontsize=13)
    ax3.tick_params(labelsize=10)

    fig.suptitle("Top Quark Study: Entanglement Entropy in the $(m_W, m_t)$ Plane",
                 fontsize=15, y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "top_w_study_combined.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_all_top_study(
    mw_grid: np.ndarray,
    mt_grid: np.ndarray,
    w_ee: np.ndarray,
    top_ee: np.ndarray,
    output_dir: str = "plots",
    show: bool = False,
) -> list[str]:
    """Generate all plots for the top/W study."""
    paths = []

    path = plot_w_ee_1d(mw_grid, mt_grid, w_ee, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    path = plot_w_ee_2d(mw_grid, mt_grid, w_ee, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    path = plot_top_ee_2d(mw_grid, mt_grid, top_ee, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    path = plot_combined(mw_grid, mt_grid, w_ee, top_ee, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    return paths
