"""Plots for Higgs EE as a function of y_t (top Yukawa) and universal kappa_f."""

import os

import matplotlib.pyplot as plt
import numpy as np

from .higgs_ee_yt import SM_MT, V_HIGGS, M_HIGGS, EE_MAX_NO_TT

SM_YT = np.sqrt(2) * SM_MT / V_HIGGS  # ~0.991


def _yt_to_mt(y):
    return y * V_HIGGS / np.sqrt(2)


def _mt_to_yt(m):
    return np.sqrt(2) * m / V_HIGGS


def plot_higgs_ee_vs_yt(results, output_dir="plots", show=False):
    """Main plot: Higgs EE (partonic + simple) vs y_t."""
    os.makedirs(output_dir, exist_ok=True)

    y = results["y_t"]
    ee_part = results["ee_partonic"]
    ee_simple = results["ee_simple"]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    valid_p = ~np.isnan(ee_part)
    valid_s = ~np.isnan(ee_simple)

    ax.plot(y[valid_p], ee_part[valid_p], color="#1f77b4", linewidth=2,
            label=r"Partonic EE ($1 - \sum P_i/N_c^i \cdot \mathrm{BR}_i^2$)")
    ax.plot(y[valid_s], ee_simple[valid_s], color="#ff7f0e", linewidth=2,
            linestyle="--",
            label=r"Simple EE ($1 - \sum \mathrm{BR}_i^2$)")

    # SM point
    sm_idx = np.argmin(np.abs(y - SM_YT))
    if valid_p[sm_idx]:
        ax.plot(SM_YT, ee_part[sm_idx], "o", color="#1f77b4", markersize=10,
                zorder=5, markeredgecolor="k", markeredgewidth=0.8,
                label=f"SM ($y_t$={SM_YT:.3f}, EE={ee_part[sm_idx]:.4f})")

    # Mark y_t = 1
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6, linewidth=1.5)
    ax.annotate(r"$y_t = 1$", xy=(1.0, ax.get_ylim()[0]),
                xytext=(1.05, 0.5), textcoords="axes fraction",
                fontsize=11, color="gray", alpha=0.8,
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    # Mark h->tt threshold
    y_tt = np.sqrt(2) * (M_HIGGS / 2) / V_HIGGS
    ax.axvline(x=y_tt, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.annotate(r"$h \to t\bar{t}$ opens", xy=(y_tt, 0.5),
                xytext=(y_tt * 0.5, 0.4),
                textcoords=("data", "axes fraction"),
                fontsize=10, color="red", alpha=0.7,
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.5))

    # Find and mark maximum EE
    if valid_p.any():
        max_idx = np.nanargmax(ee_part)
        max_y = y[max_idx]
        max_ee = ee_part[max_idx]
        ax.plot(max_y, max_ee, "*", color="#d62728", markersize=14, zorder=6,
                markeredgecolor="k", markeredgewidth=0.5,
                label=f"Max EE = {max_ee:.4f} at $y_t$ = {max_y:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel(r"$y_t$ (top Yukawa coupling)", fontsize=14)
    ax.set_ylabel("Entanglement Entropy", fontsize=14)
    ax.set_title(r"Higgs Decay Entanglement Entropy vs $y_t$", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    # Secondary axis: m_t in GeV
    ax2 = ax.secondary_xaxis("top", functions=(_yt_to_mt, _mt_to_yt))
    ax2.set_xlabel(r"$m_t$ [GeV]", fontsize=12)
    ax2.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_ee_vs_yt.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_higgs_brs_vs_yt(results, output_dir="plots", show=False):
    """Branching ratios vs y_t."""
    os.makedirs(output_dir, exist_ok=True)

    y = results["y_t"]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    channels = [
        ("br_bb", r"$b\bar{b}$", "#1f77b4"),
        ("br_WW", r"$WW^*$", "#ff7f0e"),
        ("br_gg", r"$gg$", "#2ca02c"),
        ("br_tautau", r"$\tau\tau$", "#d62728"),
        ("br_cc", r"$c\bar{c}$", "#9467bd"),
        ("br_ZZ", r"$ZZ^*$", "#8c564b"),
        ("br_tt", r"$t\bar{t}$", "#e377c2"),
        ("br_gamgam", r"$\gamma\gamma$", "#7f7f7f"),
    ]

    for key, label, color in channels:
        vals = results.get(key)
        if vals is None:
            continue
        valid = ~np.isnan(vals) & (vals > 1e-6)
        if valid.any():
            ax.plot(y[valid], vals[valid], color=color, linewidth=1.8, label=label)

    ax.axvline(x=SM_YT, color="gray", linestyle=":", alpha=0.6, linewidth=1.5)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    y_tt = np.sqrt(2) * (M_HIGGS / 2) / V_HIGGS
    ax.axvline(x=y_tt, color="red", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$y_t$ (top Yukawa coupling)", fontsize=14)
    ax.set_ylabel("Branching Ratio", fontsize=14)
    ax.set_title(r"Higgs Branching Ratios vs $y_t$", fontsize=14)
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-4)

    ax2 = ax.secondary_xaxis("top", functions=(_yt_to_mt, _mt_to_yt))
    ax2.set_xlabel(r"$m_t$ [GeV]", fontsize=12)
    ax2.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_brs_vs_yt.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_higgs_width_and_loops(results, output_dir="plots", show=False):
    """Total width ratio and loop amplitude ratios vs y_t."""
    os.makedirs(output_dir, exist_ok=True)

    y = results["y_t"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Panel 1: Total width ratio
    wr = results["width_ratio"]
    valid = ~np.isnan(wr)
    ax1.plot(y[valid], wr[valid], color="#1f77b4", linewidth=2)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax1.axvline(x=SM_YT, color="gray", linestyle=":", alpha=0.6, linewidth=1.5)
    ax1.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)
    ax1.set_ylabel(r"$\Gamma_H / \Gamma_H^{\rm SM}$", fontsize=14)
    ax1.set_title(r"Higgs Total Width and Loop Amplitudes vs $y_t$", fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Loop amplitude ratios
    R_gg = results["R_gg"]
    R_gam = results["R_gamgam"]
    valid_gg = ~np.isnan(R_gg)
    valid_gam = ~np.isnan(R_gam)

    ax2.plot(y[valid_gg], R_gg[valid_gg], color="#2ca02c", linewidth=2,
             label=r"$R_{gg} = |A_{gg}|^2 / |A_{gg}^{\rm SM}|^2$")
    ax2.plot(y[valid_gam], R_gam[valid_gam], color="#d62728", linewidth=2,
             label=r"$R_{\gamma\gamma} = |A_{\gamma\gamma}|^2 / |A_{\gamma\gamma}^{\rm SM}|^2$")

    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax2.axvline(x=SM_YT, color="gray", linestyle=":", alpha=0.6, linewidth=1.5)
    ax2.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$y_t$ (top Yukawa coupling)", fontsize=14)
    ax2.set_ylabel("Amplitude ratio", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)

    ax_top = ax1.secondary_xaxis("top", functions=(_yt_to_mt, _mt_to_yt))
    ax_top.set_xlabel(r"$m_t$ [GeV]", fontsize=12)
    ax_top.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_width_loops_vs_yt.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_higgs_ee_zoom(results, output_dir="plots", show=False):
    """Zoomed view of EE near y_t = 1 with derivatives."""
    os.makedirs(output_dir, exist_ok=True)

    y = results["y_t"]
    ee = results["ee_partonic"]
    dee = results.get("d_ee_partonic")

    # Zoom to y_t in [0.3, 5]
    mask = (y >= 0.3) & (y <= 5.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax1, ax2 = axes

    # Top panel: EE zoomed
    valid = mask & ~np.isnan(ee)
    ax1.plot(y[valid], ee[valid], color="#1f77b4", linewidth=2.5)

    sm_idx = np.argmin(np.abs(y - SM_YT))
    if not np.isnan(ee[sm_idx]):
        ax1.plot(SM_YT, ee[sm_idx], "o", color="#1f77b4", markersize=10,
                 zorder=5, markeredgecolor="k", markeredgewidth=0.8,
                 label=f"SM ($y_t$={SM_YT:.3f})")

    # Find max in zoom region
    ee_zoom = np.where(mask & ~np.isnan(ee), ee, -np.inf)
    max_idx = np.argmax(ee_zoom)
    ax1.plot(y[max_idx], ee[max_idx], "*", color="#d62728", markersize=14,
             zorder=6, markeredgecolor="k", markeredgewidth=0.5,
             label=f"Max EE = {ee[max_idx]:.6f} at $y_t$ = {y[max_idx]:.3f}")

    ax1.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6, linewidth=1.5,
                label=r"$y_t = 1$")
    ax1.axvline(x=SM_YT, color="gray", linestyle="--", alpha=0.3)

    y_tt = np.sqrt(2) * (M_HIGGS / 2) / V_HIGGS
    ax1.axvline(x=y_tt, color="red", linestyle="--", alpha=0.5, linewidth=1,
                label=r"$h\to t\bar{t}$ threshold")

    ax1.set_ylabel("Partonic EE", fontsize=14)
    ax1.set_title(r"Higgs EE near $y_t = 1$ (zoomed)", fontsize=14)
    ax1.legend(fontsize=10, loc="best")
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: derivative
    if dee is not None:
        valid_d = mask & ~np.isnan(dee)
        ax2.plot(y[valid_d], dee[valid_d], color="#2ca02c", linewidth=2)
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax2.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6, linewidth=1.5)
        ax2.axvline(x=SM_YT, color="gray", linestyle="--", alpha=0.3)
        ax2.axvline(x=y_tt, color="red", linestyle="--", alpha=0.5, linewidth=1)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$y_t$", fontsize=14)
    ax2.set_ylabel(r"$d\,\mathrm{EE}/d\ln y_t$", fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)

    ax_top = ax1.secondary_xaxis("top", functions=(_yt_to_mt, _mt_to_yt))
    ax_top.set_xlabel(r"$m_t$ [GeV]", fontsize=12)
    ax_top.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_ee_zoom_yt1.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_higgs_shannon_vs_yt(results, output_dir="plots", show=False):
    """Shannon entropy and effective number of channels vs y_t."""
    os.makedirs(output_dir, exist_ok=True)

    y = results["y_t"]
    shan = results["shannon"]
    ee_s = results["ee_simple"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Shannon
    valid = ~np.isnan(shan)
    ax1.plot(y[valid], shan[valid], color="#9467bd", linewidth=2)
    sm_idx = np.argmin(np.abs(y - SM_YT))
    if not np.isnan(shan[sm_idx]):
        ax1.plot(SM_YT, shan[sm_idx], "o", color="#9467bd", markersize=8,
                 zorder=5, markeredgecolor="k", markeredgewidth=0.8)
    ax1.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6)
    ax1.set_ylabel("Shannon Entropy", fontsize=14)
    ax1.set_title(r"Entropy Measures vs $y_t$", fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)

    # Effective number of channels: n_eff = 1/(Sum BR^2) = 1/(1 - EE_simple)
    valid_s = ~np.isnan(ee_s) & (ee_s < 1.0)
    n_eff = 1.0 / (1.0 - ee_s[valid_s])
    ax2.plot(y[valid_s], n_eff, color="#e377c2", linewidth=2)
    if not np.isnan(ee_s[sm_idx]):
        n_eff_sm = 1.0 / (1.0 - ee_s[sm_idx])
        ax2.plot(SM_YT, n_eff_sm, "o", color="#e377c2", markersize=8,
                 zorder=5, markeredgecolor="k", markeredgewidth=0.8)
    ax2.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$y_t$", fontsize=14)
    ax2.set_ylabel(r"$n_{\rm eff} = 1/\sum \mathrm{BR}_i^2$", fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)

    ax_top = ax1.secondary_xaxis("top", functions=(_yt_to_mt, _mt_to_yt))
    ax_top.set_xlabel(r"$m_t$ [GeV]", fontsize=12)
    ax_top.tick_params(labelsize=10)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_shannon_neff_vs_yt.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_kf_ee(results_kf, output_dir="plots", show=False):
    """Main result: Higgs EE vs universal kappa_f showing maximum at SM."""
    os.makedirs(output_dir, exist_ok=True)

    kf = results_kf["kappa_f"]
    ee = results_kf["ee_partonic"]
    dee = results_kf.get("d_ee_partonic")

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1]})
    ax1, ax2 = axes

    # ── Top panel: EE vs kf ──
    valid = ~np.isnan(ee)
    ax1.plot(kf[valid], ee[valid], color="#1f77b4", linewidth=2.5,
             label=r"$\mathrm{EE}(\kappa_f)$")

    # EE_max line
    ax1.axhline(y=EE_MAX_NO_TT, color="#2ca02c", linestyle="--", alpha=0.5,
                linewidth=1.5, label=f"$\\mathrm{{EE}}_{{\\max}}$ = {EE_MAX_NO_TT:.4f}")

    # SM point
    sm_idx = np.argmin(np.abs(kf - 1.0))
    ax1.plot(1.0, ee[sm_idx], "o", color="#1f77b4", markersize=12,
             zorder=10, markeredgecolor="k", markeredgewidth=1.0,
             label=f"SM ($\\kappa_f$=1, EE={ee[sm_idx]:.4f})")

    # Find and mark maximum
    max_idx = np.nanargmax(ee)
    ax1.plot(kf[max_idx], ee[max_idx], "*", color="#d62728", markersize=16,
             zorder=11, markeredgecolor="k", markeredgewidth=0.5,
             label=(f"Max EE = {ee[max_idx]:.6f}\n"
                    f"  at $\\kappa_f$ = {kf[max_idx]:.3f}"))

    ax1.axvline(x=1.0, color="gray", linestyle=":", alpha=0.4, linewidth=1)

    ax1.set_ylabel("Partonic Entanglement Entropy", fontsize=14)
    ax1.set_title(
        r"Higgs Decay EE vs Universal Fermion Coupling $\kappa_f$"
        "\n(SM maximizes entanglement entropy)",
        fontsize=14)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)

    # ── Bottom panel: derivative ──
    if dee is not None:
        valid_d = ~np.isnan(dee)
        ax2.plot(kf[valid_d], dee[valid_d], color="#2ca02c", linewidth=2)
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax2.axvline(x=1.0, color="gray", linestyle=":", alpha=0.4)

        # Mark zero crossing
        signs = np.sign(dee[valid_d])
        kf_d = kf[valid_d]
        dee_d = dee[valid_d]
        zcs = np.where(np.diff(signs))[0]
        for zc in zcs:
            y_interp = kf_d[zc] + (kf_d[zc+1] - kf_d[zc]) * abs(dee_d[zc]) / (abs(dee_d[zc]) + abs(dee_d[zc+1]))
            ax2.plot(y_interp, 0, "o", color="#d62728", markersize=8, zorder=5,
                     markeredgecolor="k", markeredgewidth=0.5)
            ax2.annotate(f"$\\kappa_f$ = {y_interp:.3f}",
                         xy=(y_interp, 0), xytext=(y_interp * 1.3, 0.002),
                         fontsize=10, color="#d62728",
                         arrowprops=dict(arrowstyle="->", color="#d62728"))

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\kappa_f$ (universal fermion coupling rescaling)", fontsize=14)
    ax2.set_ylabel(r"$d\,\mathrm{EE}/d\ln\kappa_f$", fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_ee_vs_kappa_f.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_kf_brs(results_kf, output_dir="plots", show=False):
    """Branching ratios vs kappa_f."""
    os.makedirs(output_dir, exist_ok=True)

    kf = results_kf["kappa_f"]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    channels = [
        ("br_bb", r"$b\bar{b}$", "#1f77b4"),
        ("br_WW", r"$WW^*$", "#ff7f0e"),
        ("br_gg", r"$gg$", "#2ca02c"),
        ("br_tautau", r"$\tau\tau$", "#d62728"),
        ("br_cc", r"$c\bar{c}$", "#9467bd"),
        ("br_ZZ", r"$ZZ^*$", "#8c564b"),
        ("br_gamgam", r"$\gamma\gamma$", "#7f7f7f"),
    ]

    for key, label, color in channels:
        vals = results_kf.get(key)
        if vals is None:
            continue
        valid = ~np.isnan(vals) & (vals > 1e-6)
        if valid.any():
            ax.plot(kf[valid], vals[valid], color=color, linewidth=1.8, label=label)

    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6, linewidth=1.5,
               label="SM")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa_f$", fontsize=14)
    ax.set_ylabel("Branching Ratio", fontsize=14)
    ax.set_title(r"Higgs Branching Ratios vs $\kappa_f$", fontsize=14)
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1e-4)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_brs_vs_kappa_f.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_kf_ee_decomposition(results_kf, output_dir="plots", show=False):
    """Show how the EE maximum arises from the competition between channels."""
    os.makedirs(output_dir, exist_ok=True)

    kf = results_kf["kappa_f"]
    ee = results_kf["ee_partonic"]
    br_bb = results_kf["br_bb"]
    br_WW = results_kf["br_WW"]
    br_gg = results_kf["br_gg"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Top: EE and individual purity contributions
    valid = ~np.isnan(ee)

    # Purity contributions (what makes EE < 1)
    from .higgs_ee_yt import SPIN_FACTORS, COLOR_FACTORS
    pur_bb = np.where(valid, (SPIN_FACTORS["bb"] / COLOR_FACTORS["bb"]) * br_bb**2, np.nan)
    pur_WW = np.where(valid, (SPIN_FACTORS["WW"] / COLOR_FACTORS["WW"]) * br_WW**2, np.nan)
    pur_gg = np.where(valid, (SPIN_FACTORS["gg"] / COLOR_FACTORS["gg"]) * br_gg**2, np.nan)

    ax1.plot(kf[valid], ee[valid], "k-", linewidth=2.5, label="EE (total)")
    ax1.axhline(y=EE_MAX_NO_TT, color="gray", linestyle="--", alpha=0.4)
    ax1.plot(1.0, ee[np.argmin(np.abs(kf - 1.0))], "o", color="k",
             markersize=10, zorder=10, markeredgecolor="k", markeredgewidth=0.8)

    ax1.set_ylabel("Partonic EE", fontsize=14)
    ax1.set_title(
        r"EE decomposition: competition between $b\bar{b}$, $WW^*$, and $gg$",
        fontsize=14)
    ax1.legend(fontsize=11, loc="lower right")
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)

    # Bottom: purity contributions (stacked area style)
    ax2.fill_between(kf[valid], 0, pur_bb[valid],
                     color="#1f77b4", alpha=0.6, label=r"$b\bar{b}$ purity")
    ax2.fill_between(kf[valid], pur_bb[valid], pur_bb[valid] + pur_WW[valid],
                     color="#ff7f0e", alpha=0.6, label=r"$WW^*$ purity")
    ax2.fill_between(kf[valid], pur_bb[valid] + pur_WW[valid],
                     pur_bb[valid] + pur_WW[valid] + pur_gg[valid],
                     color="#2ca02c", alpha=0.6, label=r"$gg$ purity")
    ax2.plot(kf[valid], 1.0 - ee[valid], "k-", linewidth=1.5,
             label="Total purity (= 1 - EE)")

    ax2.axvline(x=1.0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\kappa_f$", fontsize=14)
    ax2.set_ylabel(r"Purity = $\sum (P_i/N_c^i)\,\mathrm{BR}_i^2$", fontsize=14)
    ax2.legend(fontsize=10, loc="best")
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_ee_decomposition_kf.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_all_higgs_ee(results, output_dir="plots", show=False,
                      results_kf=None):
    """Generate all Higgs EE plots."""
    paths = []

    path = plot_higgs_ee_vs_yt(results, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    path = plot_higgs_ee_zoom(results, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    path = plot_higgs_brs_vs_yt(results, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    path = plot_higgs_width_and_loops(results, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    path = plot_higgs_shannon_vs_yt(results, output_dir, show)
    paths.append(path)
    print(f"  Saved: {path}")

    # Universal kappa_f plots
    if results_kf is not None:
        path = plot_kf_ee(results_kf, output_dir, show)
        paths.append(path)
        print(f"  Saved: {path}")

        path = plot_kf_brs(results_kf, output_dir, show)
        paths.append(path)
        print(f"  Saved: {path}")

        path = plot_kf_ee_decomposition(results_kf, output_dir, show)
        paths.append(path)
        print(f"  Saved: {path}")

    return paths
