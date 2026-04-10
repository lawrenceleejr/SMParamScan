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


def plot_joint_ee(
    mw_grid: np.ndarray,
    mt_grid: np.ndarray,
    w_ee: np.ndarray,
    top_ee: np.ndarray,
    output_dir: str = "plots",
    show: bool = False,
) -> str:
    """Single 2D heatmap of joint EE = EE_W * EE_top in the (m_W, m_t) plane."""
    os.makedirs(output_dir, exist_ok=True)

    joint_ee = w_ee * top_ee

    # Find maximum
    valid = ~np.isnan(joint_ee)
    max_idx = np.unravel_index(np.nanargmax(joint_ee), joint_ee.shape)
    max_mt = mt_grid[max_idx[0]]
    max_mw = mw_grid[max_idx[1]]
    max_val = joint_ee[max_idx]

    # SM point value
    sm_mt_idx = np.argmin(np.abs(mt_grid - SM_MT))
    sm_mw_idx = np.argmin(np.abs(mw_grid - SM_MW))
    sm_val = joint_ee[sm_mt_idx, sm_mw_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    MW, MT = np.meshgrid(mw_grid, mt_grid)
    im = ax.pcolormesh(MW, MT, joint_ee, cmap="inferno", shading="gouraud")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\mathrm{EE}_W \times \mathrm{EE}_t$", fontsize=14)

    # Contour lines
    if valid.any():
        ee_min = np.nanmin(joint_ee[valid])
        ee_max = np.nanmax(joint_ee[valid])
        levels = np.linspace(ee_min, ee_max, 15)
        cs = ax.contour(MW, MT, joint_ee, levels=levels, colors="w",
                        linewidths=0.5, alpha=0.5)
        ax.clabel(cs, fontsize=7, fmt="%.3f")

    # On-shell W threshold for top: m_t = m_W + m_b
    mw_line = np.linspace(mw_grid[0], mw_grid[-1], 300)
    mt_onshell = mw_line + M_B
    mask = (mt_onshell >= mt_grid[0]) & (mt_onshell <= mt_grid[-1])
    ax.plot(mw_line[mask], mt_onshell[mask], "--", color="cyan", linewidth=1.5,
            alpha=0.8, label=r"$m_t = m_W + m_b$")

    # tb threshold for W: m_W = m_t + m_b
    mt_line = np.linspace(mt_grid[0], mt_grid[-1], 300)
    mw_tb = mt_line + M_B
    mask2 = (mw_tb >= mw_grid[0]) & (mw_tb <= mw_grid[-1])
    ax.plot(mw_tb[mask2], mt_line[mask2], ":", color="cyan", linewidth=1.5,
            alpha=0.8, label=r"$m_W = m_t + m_b$")

    # Maximum
    ax.plot(max_mw, max_mt, "*", color="lime", markersize=16, zorder=10,
            markeredgecolor="k", markeredgewidth=1.0,
            label=(f"Max = {max_val:.3f}\n"
                   f"  ($m_W$={max_mw:.1f}, $m_t$={max_mt:.1f} GeV)"))

    # SM point
    ax.plot(SM_MW, SM_MT, "o", color="white", markersize=10, zorder=10,
            markeredgecolor="k", markeredgewidth=1.0,
            label=f"SM = {sm_val:.3f}\n  ($m_W$=80.4, $m_t$=172.5 GeV)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_W$ [GeV]", fontsize=14)
    ax.set_ylabel(r"$m_t$ [GeV]", fontsize=14)
    ax.set_title(
        r"Joint Entanglement Entropy: $\mathrm{EE}_W \times \mathrm{EE}_t$"
        "\nin the $(m_W, m_t)$ plane",
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="upper left",
              framealpha=0.9, edgecolor="gray")
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    path = os.path.join(output_dir, "joint_ee_top_w.pdf")
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

    # Primary output: joint EE product
    path = plot_joint_ee(mw_grid, mt_grid, w_ee, top_ee, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    # Supporting plots
    path = plot_w_ee_1d(mw_grid, mt_grid, w_ee, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    path = plot_w_ee_2d(mw_grid, mt_grid, w_ee, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    path = plot_top_ee_2d(mw_grid, mt_grid, top_ee, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    return paths
