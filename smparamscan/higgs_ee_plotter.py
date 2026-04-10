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


def plot_kf_three_regimes(results_kf, output_dir="plots", show=False):
    """Compare partonic, hadronized, and inclusive EE vs kappa_f.

    Key plot: shows that the three regimes peak at different kappa_f,
    revealing the tension between the top (partonic) and other quarks
    (hadronized).
    """
    os.makedirs(output_dir, exist_ok=True)

    kf = results_kf["kappa_f"]
    ee_p = results_kf["ee_partonic"]
    ee_h = results_kf["ee_hadronized"]
    ee_i = results_kf["ee_inclusive"]
    hcost = results_kf.get("hadronization_cost")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2.5, 1]})

    # ── Top panel: three EE regimes ──
    valid_p = ~np.isnan(ee_p)
    valid_h = ~np.isnan(ee_h)
    valid_i = ~np.isnan(ee_i)

    ax1.plot(kf[valid_p], ee_p[valid_p], color="#1f77b4", linewidth=2.5,
             label="Partonic (full color)")
    ax1.plot(kf[valid_h], ee_h[valid_h], color="#d62728", linewidth=2.5,
             label="Hadronized ($N_c \\to 1$)")
    ax1.plot(kf[valid_i], ee_i[valid_i], color="#ff7f0e", linewidth=2,
             linestyle="--", label="Inclusive (merge light hadrons)")

    # Mark maxima
    for ee, color, label, regime in [
        (ee_p, "#1f77b4", "partonic", "Partonic"),
        (ee_h, "#d62728", "hadronized", "Hadronized"),
    ]:
        max_idx = np.nanargmax(ee)
        ax1.plot(kf[max_idx], ee[max_idx], "*", color=color, markersize=14,
                 zorder=10, markeredgecolor="k", markeredgewidth=0.5)
        ax1.annotate(f"{regime} max\n$\\kappa_f$={kf[max_idx]:.2f}",
                     xy=(kf[max_idx], ee[max_idx]),
                     xytext=(kf[max_idx] * 1.8, ee[max_idx] - 0.02),
                     fontsize=9, color=color,
                     arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))

    # SM point
    sm_idx = np.argmin(np.abs(kf - 1.0))
    ax1.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.plot(1.0, ee_p[sm_idx], "o", color="#1f77b4", markersize=10,
             zorder=11, markeredgecolor="k", markeredgewidth=0.8)
    ax1.plot(1.0, ee_h[sm_idx], "s", color="#d62728", markersize=8,
             zorder=11, markeredgecolor="k", markeredgewidth=0.8)

    # Shade the hadronization cost region
    ax1.fill_between(kf[valid_p & valid_h], ee_h[valid_p & valid_h],
                     ee_p[valid_p & valid_h], alpha=0.12, color="gray",
                     label="Hadronization cost")

    ax1.set_ylabel("Entanglement Entropy", fontsize=14)
    ax1.set_title(
        r"Higgs EE: partonic vs hadronized regimes"
        "\n(top quark sees partonic; light quarks see hadronized)",
        fontsize=13)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)

    # ── Bottom panel: hadronization cost ──
    if hcost is not None:
        valid_hc = ~np.isnan(hcost)
        ax2.plot(kf[valid_hc], hcost[valid_hc], color="#2ca02c", linewidth=2.5)
        ax2.fill_between(kf[valid_hc], 0, hcost[valid_hc],
                         alpha=0.2, color="#2ca02c")
        ax2.plot(1.0, hcost[sm_idx], "o", color="#2ca02c", markersize=8,
                 zorder=5, markeredgecolor="k", markeredgewidth=0.8,
                 label=f"SM: $\\Delta$EE = {hcost[sm_idx]:.3f}")
        ax2.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
        ax2.legend(fontsize=11, loc="upper left")

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\kappa_f$ (universal fermion coupling)", fontsize=14)
    ax2.set_ylabel(r"$\Delta\mathrm{EE}$ (hadronization cost)", fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_ee_three_regimes.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_individual_kq_scans(quark_results, output_dir="plots", show=False):
    """Plot EE in all three regimes for each quark's individual kappa_q scan.

    quark_results: dict of quark_name -> scan results dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    quarks = list(quark_results.keys())
    n = len(quarks)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    quark_labels = {"t": "top", "b": "bottom", "c": "charm", "s": "strange"}
    quark_sym = {"t": r"$\kappa_t$", "b": r"$\kappa_b$",
                 "c": r"$\kappa_c$", "s": r"$\kappa_s$"}

    for idx, quark in enumerate(quarks[:4]):
        ax = axes[idx]
        res = quark_results[quark]
        kq = res["kappa_q"]
        ee_p = res["ee_partonic"]
        ee_h = res["ee_hadronized"]
        ee_i = res["ee_inclusive"]

        valid_p = ~np.isnan(ee_p)
        valid_h = ~np.isnan(ee_h)

        ax.plot(kq[valid_p], ee_p[valid_p], color="#1f77b4", linewidth=2,
                label="Partonic")
        ax.plot(kq[valid_h], ee_h[valid_h], color="#d62728", linewidth=2,
                label="Hadronized")

        # Mark maxima
        max_p = np.nanargmax(ee_p)
        max_h = np.nanargmax(ee_h)
        ax.plot(kq[max_p], ee_p[max_p], "*", color="#1f77b4", markersize=12,
                zorder=10, markeredgecolor="k", markeredgewidth=0.5)
        ax.plot(kq[max_h], ee_h[max_h], "*", color="#d62728", markersize=12,
                zorder=10, markeredgecolor="k", markeredgewidth=0.5)

        # SM point
        sm_idx = np.argmin(np.abs(kq - 1.0))
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.plot(1.0, ee_p[sm_idx], "o", color="#1f77b4", markersize=7,
                zorder=11, markeredgecolor="k", markeredgewidth=0.6)
        ax.plot(1.0, ee_h[sm_idx], "s", color="#d62728", markersize=6,
                zorder=11, markeredgecolor="k", markeredgewidth=0.6)

        # Shade hadronization cost
        ax.fill_between(kq[valid_p & valid_h], ee_h[valid_p & valid_h],
                         ee_p[valid_p & valid_h], alpha=0.1, color="gray")

        label_str = quark_labels.get(quark, quark)
        ax.set_title(f"{label_str} ({quark_sym[quark]}): "
                     f"part. max @ {kq[max_p]:.2f}, hadr. max @ {kq[max_h]:.2f}",
                     fontsize=11)
        ax.set_xscale("log")
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)
        if idx >= 2:
            ax.set_xlabel(r"$\kappa_q$", fontsize=12)
        if idx % 2 == 0:
            ax.set_ylabel("EE", fontsize=12)
        if idx == 0:
            ax.legend(fontsize=9, loc="best")

    fig.suptitle("Higgs EE vs individual quark coupling: partonic vs hadronized",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_ee_individual_quarks.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_ee_vs_multiplicity(mult_results, output_dir="plots", show=False):
    """Plot EE as a function of hadronic fragmentation multiplicity.

    Shows the crossover: at low M, hadronization hurts EE (naive model);
    at M > N_c, hadronization helps EE (multiplicity wins over color loss).
    """
    os.makedirs(output_dir, exist_ok=True)

    M_q = mult_results["M_quarks"]
    ee_frag = mult_results["ee_fragmented"]
    ee_part = mult_results["ee_partonic"]
    ee_had = mult_results["ee_naive_hadronized"]

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(M_q, ee_frag, color="#1f77b4", linewidth=2.5,
            label="Fragmented EE (multiplicity model)")
    ax.axhline(ee_part[0], color="#2ca02c", linewidth=2, linestyle="--",
               label=f"Partonic EE = {ee_part[0]:.4f}")
    ax.axhline(ee_had[0], color="#d62728", linewidth=2, linestyle=":",
               label=f"Naive hadronized (M=1) = {ee_had[0]:.4f}")

    # Mark crossover at M_q = N_c = 3 (where fragmented = partonic)
    ax.axvline(x=3.0, color="#2ca02c", alpha=0.5, linestyle="-.",
               label=r"$M_q = N_c = 3$ (crossover)")
    ax.axvline(x=1.0, color="#d62728", alpha=0.5, linestyle="-.",
               label=r"$M_q = 1$ (naive hadronized)")

    # Shade regions
    ax.fill_between(M_q[M_q <= 3.0], ee_had[0], ee_part[0],
                    alpha=0.08, color="#d62728",
                    label="Hadronization reduces EE")
    ax.fill_between(M_q[M_q >= 3.0], ee_part[0],
                    np.maximum(ee_frag[M_q >= 3.0], ee_part[0]),
                    alpha=0.08, color="#2ca02c",
                    label="Hadronization increases EE")

    # Mark realistic multiplicity range (~10-50 for quarks at Higgs scale)
    ax.axvspan(10, 50, alpha=0.12, color="#ff7f0e", zorder=0,
               label="Realistic $M_q$ range\n(~10-50 at Higgs scale)")

    # Compute and annotate EE at realistic M
    M_realistic = 20
    idx_r = np.argmin(np.abs(M_q - M_realistic))
    ee_r = ee_frag[idx_r]
    ax.plot(M_realistic, ee_r, "D", color="#ff7f0e", markersize=10,
            zorder=10, markeredgecolor="k", markeredgewidth=0.8)
    ax.annotate(f"$M_q$=20: EE={ee_r:.4f}\n({(ee_r-ee_part[0])/ee_part[0]*100:+.1f}% vs partonic)",
                xy=(M_realistic, ee_r),
                xytext=(M_realistic * 2.5, ee_r - 0.01),
                fontsize=10, color="#ff7f0e",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e"))

    ax.set_xscale("log")
    ax.set_xlabel(r"Effective hadronic multiplicity $M_q$ (quarks)", fontsize=14)
    ax.set_ylabel("Entanglement Entropy", fontsize=14)
    ax.set_title(
        "Higgs EE vs hadronic fragmentation multiplicity (SM branching ratios)\n"
        r"$M_g = \frac{9}{4} M_q$ (QCD Casimir ratio); "
        "crossover at $M_q = N_c = 3$",
        fontsize=12)
    ax.legend(fontsize=9, loc="lower right", ncol=1)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_ee_vs_multiplicity.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_kf_with_multiplicity(kf_mult_results, output_dir="plots", show=False):
    """Plot kappa_f scan at several hadronic multiplicity values.

    Shows how the optimal kappa_f shifts as multiplicity increases:
    more multiplicity -> hadronization helps -> optimum moves toward
    maximizing colored-channel fraction.
    """
    os.makedirs(output_dir, exist_ok=True)

    kf = kf_mult_results["kappa_f"]
    ee_part = kf_mult_results["ee_partonic"]
    M_vals = kf_mult_results["M_values_quarks"]

    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot partonic first
    valid = ~np.isnan(ee_part)
    ax.plot(kf[valid], ee_part[valid], color="black", linewidth=2.5,
            linestyle="--", label="Partonic (full color)", zorder=5)

    # Color gradient for multiplicity values
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(M_vals)))

    for j, M_q in enumerate(M_vals):
        key = f"ee_M{M_q:.0f}"
        ee = kf_mult_results[key]
        valid_m = ~np.isnan(ee)
        lbl = f"$M_q$ = {M_q:.0f}"
        if M_q == 1:
            lbl += " (naive hadr.)"
        elif M_q == 3:
            lbl += " ($= N_c$, crossover)"
        ls = "-" if M_q >= 3 else ":"
        ax.plot(kf[valid_m], ee[valid_m], color=colors[j], linewidth=2,
                linestyle=ls, label=lbl)

        # Mark maximum
        max_idx = np.nanargmax(ee)
        ax.plot(kf[max_idx], ee[max_idx], "*", color=colors[j],
                markersize=12, zorder=10, markeredgecolor="k",
                markeredgewidth=0.5)

    # SM point
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5,
               label="SM ($\\kappa_f = 1$)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_f$ (universal fermion coupling)", fontsize=14)
    ax.set_ylabel("Entanglement Entropy", fontsize=14)
    ax.set_title(
        "Higgs EE vs $\\kappa_f$ at different hadronic multiplicities\n"
        "Higher multiplicity $\\to$ hadronization increases EE $\\to$ "
        "optimum shifts",
        fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "higgs_ee_kf_multiplicity.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_all_higgs_ee(results, output_dir="plots", show=False,
                      results_kf=None, quark_results=None,
                      mult_results=None, kf_mult_results=None):
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

        path = plot_kf_three_regimes(results_kf, output_dir, show)
        paths.append(path)
        print(f"  Saved: {path}")

    # Individual quark scans
    if quark_results is not None:
        path = plot_individual_kq_scans(quark_results, output_dir, show)
        paths.append(path)
        print(f"  Saved: {path}")

    # Fragmentation multiplicity plots
    if mult_results is not None:
        path = plot_ee_vs_multiplicity(mult_results, output_dir, show)
        paths.append(path)
        print(f"  Saved: {path}")

    if kf_mult_results is not None:
        path = plot_kf_with_multiplicity(kf_mult_results, output_dir, show)
        paths.append(path)
        print(f"  Saved: {path}")

    return paths
