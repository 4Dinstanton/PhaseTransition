#!/usr/bin/env python3
"""
Gravitational wave spectrum from a first-order cosmological phase transition.

Reads tunneling CSV data (T, S3/T), computes nucleation parameters
(T_n, T_RH, beta/H, HR_*, alpha), and plots the GW power spectrum with
detector sensitivity curves overlaid.

Sound wave contribution follows Hindmarsh et al. (2017) as parametrized in
Dutka, Jung, and Shin (2024) [arXiv:2412.15864], eqs. (4.31)-(4.33).
Suppression factor from Ellis, Lewicki, and No (2019) [arXiv:1809.08242].

Usage (from project root):
    python analysis/gwSpectrum.py data/tunneling/set6/T-S_param_set6_lambdaSix_0E+00_fermion_only.csv
    python analysis/gwSpectrum.py <csv_path> --vw 0.9 --delV 1e28 --g_star 106.75
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
M_PL = 2.4e18  # reduced Planck mass (GeV)
G_STAR_DEFAULT = 106.75  # SM relativistic DOF
CHI_G2 = 30.0 / (math.pi**2 * G_STAR_DEFAULT)
DEL_V_DEFAULT = 1.0e28  # vacuum energy difference (GeV^4)
H_PARAM = 0.674  # h = H0 / (100 km/s/Mpc)


# ---------------------------------------------------------------------------
# Hubble parameter
# ---------------------------------------------------------------------------
def hubble(T, del_V=DEL_V_DEFAULT):
    """H(T) in GeV, radiation + vacuum energy."""
    T = np.asarray(T, dtype=float)
    return np.sqrt((T**4 / CHI_G2 + del_V) / (3.0 * M_PL**2))


# ---------------------------------------------------------------------------
# Reheating temperature: rho_rad(T_RH) = delV
# ---------------------------------------------------------------------------
def compute_T_RH(del_V, g_star):
    """T_RH = (30 delV / (pi^2 g_*))^{1/4}."""
    return (30.0 * del_V / (math.pi**2 * g_star)) ** 0.25


def compute_T_RH_model(Ms, gX, mu_star, g_star):
    """T_RH from model parameters. Eq. (4.9) of arXiv:2412.15864."""
    return (
        1.6e6  # 1.6 PeV in GeV
        * (100.0 / g_star) ** 0.25
        * (math.sqrt(gX) * Ms / 1.0e4) ** 0.5
        * (mu_star / 1.0e10) ** 0.5
    )


def compute_delV_model(Ms, gX, mu_star):
    """delV from model parameters. Eq. (4.7) of arXiv:2412.15864."""
    return math.e**0.5 / (8.0 * math.pi**2) * gX * Ms**2 * mu_star**2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_tunneling_data(csv_path):
    """Return (T, S3_over_T) arrays sorted by ascending T."""
    df = pd.read_csv(csv_path)
    if "S3/T" not in df.columns or "T" not in df.columns:
        raise ValueError(
            f"CSV must contain 'T' and 'S3/T' columns. Found: {list(df.columns)}"
        )
    df = df.dropna(subset=["T", "S3/T"])
    df = df[df["S3/T"] > 0].copy()
    df.sort_values("T", inplace=True)
    return df["T"].values, df["S3/T"].values


# ---------------------------------------------------------------------------
# Nucleation rate  ln(Gamma) = -S3/T + 4 ln T + (3/2) ln(S3/(2piT))
# ---------------------------------------------------------------------------
def ln_gamma(T, S3_T):
    T = np.asarray(T, dtype=float)
    S3_T = np.asarray(S3_T, dtype=float)
    return -S3_T + 4.0 * np.log(T) + 1.5 * np.log(S3_T / (2.0 * math.pi))


# ---------------------------------------------------------------------------
# Find nucleation temperature  Gamma / H^4 = 1
# ---------------------------------------------------------------------------
def find_nucleation_temp(T, S3_T, del_V=DEL_V_DEFAULT):
    lg = ln_gamma(T, S3_T)
    H = hubble(T, del_V)
    log_ratio = lg - 4.0 * np.log(H)

    crossings = np.where(np.diff(np.sign(log_ratio)))[0]
    if len(crossings) == 0:
        max_ratio = np.max(log_ratio)
        idx_max = np.argmax(log_ratio)
        print(
            f"  WARNING: Gamma/H^4 never reaches 1.  "
            f"Max ln(Gamma/H^4) = {max_ratio:.2f} at T = {T[idx_max]:.1f} GeV"
        )
        if max_ratio > -5:
            print("  Using temperature of maximum Gamma/H^4 as approximate T_n")
            T_n = T[idx_max]
        else:
            raise RuntimeError("Nucleation condition Gamma/H^4 >= 1 never met.")
    else:
        idx = crossings[0]
        t0 = log_ratio[idx] / (log_ratio[idx] - log_ratio[idx + 1])
        T_n = T[idx] + t0 * (T[idx + 1] - T[idx])
    return T_n


# ---------------------------------------------------------------------------
# beta/H = -d ln(Gamma) / d ln(T)  at T_n
# ---------------------------------------------------------------------------
def compute_beta_over_H(T, S3_T, T_n):
    lg = ln_gamma(T, S3_T)
    ln_T = np.log(T)
    cs = CubicSpline(ln_T, lg)
    dln_Gamma_dln_T = cs(np.log(T_n), 1)
    return float(-dln_Gamma_dln_T)


# ---------------------------------------------------------------------------
# Transition strength  alpha = delV / rho_rad(T_n)
# ---------------------------------------------------------------------------
def compute_alpha(T_n, del_V, g_star):
    rho_rad = (math.pi**2 / 30.0) * g_star * T_n**4
    return del_V / rho_rad


# ---------------------------------------------------------------------------
# Mean bubble separation  HR_* = (8pi)^{1/3} / (beta/H)
# ---------------------------------------------------------------------------
def compute_HR_star(beta_H):
    return (8.0 * math.pi) ** (1.0 / 3.0) / beta_H


# ---------------------------------------------------------------------------
# RMS fluid velocity  Uf^2 = (3/4) kappa_v alpha / (1 + alpha)
# ---------------------------------------------------------------------------
def compute_Uf(kv, alpha_val):
    return math.sqrt(0.75 * kv * alpha_val / (1.0 + alpha_val))


# ---------------------------------------------------------------------------
# Efficiency factors (Espinosa et al. 2010)
# ---------------------------------------------------------------------------
def kappa_v(alpha):
    return alpha / (0.73 + 0.083 * math.sqrt(alpha) + alpha)


def kappa_turb_frac():
    return 0.05


# ---------------------------------------------------------------------------
# Peak frequencies (Hz today)
# ---------------------------------------------------------------------------
def f_peak_sw(HR_star, T_RH, g_star, z_p=10.0):
    """Sound wave peak. Eq. (4.33) of arXiv:2412.15864."""
    return (
        8.9
        * (8.0 * math.pi) ** (1.0 / 3.0)
        * 1.0e-2
        / HR_star
        * (z_p / 10.0)
        * (T_RH / 1.0e6)
        * (g_star / 100.0) ** (1.0 / 6.0)
    )


def f_peak_env(beta_H, T_RH, g_star, v_w):
    """Envelope peak. Caprini et al. (2016)."""
    return (
        1.65e-5
        * (T_RH / 100.0)
        * (g_star / 100.0) ** (1.0 / 6.0)
        * beta_H
        * 0.62
        / (1.8 - 0.1 * v_w + v_w**2)
    )


def f_peak_turb(beta_H, T_RH, g_star, v_w):
    """Turbulence peak. Caprini et al. (2016)."""
    return (
        2.7e-5 * (1.0 / v_w) * beta_H * (T_RH / 100.0) * (g_star / 100.0) ** (1.0 / 6.0)
    )


# ---------------------------------------------------------------------------
# GW spectrum: sound waves
#   Eq. (4.31) of arXiv:2412.15864  (from Hindmarsh et al. 2017)
# ---------------------------------------------------------------------------
def gw_sound_wave(f, HR_star, Uf, g_star, fp):
    """h^2 Omega_GW from sound waves."""
    f = np.asarray(f, dtype=float)
    F_GW0 = 3.5e-5 * (100.0 / g_star) ** (1.0 / 3.0)
    Gamma = 4.0 / 3.0
    Omega_tilde = 1.0e-2
    x = f / fp
    S_sw = x**3 * (7.0 / (4.0 + 3.0 * x**2)) ** 3.5
    return 2.061 * F_GW0 * H_PARAM**2 * Gamma**2 * Uf**4 * HR_star * Omega_tilde * S_sw


# ---------------------------------------------------------------------------
# GW spectrum: bubble collision (envelope)  -- Caprini et al. (2016)
# ---------------------------------------------------------------------------
def gw_envelope(f, alpha_val, beta_H, g_star, kphi, fp):
    f = np.asarray(f, dtype=float)
    Sf = 3.8 * (f / fp) ** 2.8 / (1.0 + 2.8 * (f / fp) ** 3.8)
    return (
        1.67e-5
        * (1.0 / beta_H) ** 2
        * (kphi * alpha_val / (1.0 + alpha_val)) ** 2
        * (100.0 / g_star) ** (1.0 / 3.0)
        * Sf
    )


# ---------------------------------------------------------------------------
# GW spectrum: MHD turbulence  -- Caprini et al. (2009)
# ---------------------------------------------------------------------------
def gw_turbulence(f, alpha_val, beta_H, T_RH, g_star, v_w, kturb, fp):
    f = np.asarray(f, dtype=float)
    h_star = 1.65e-5 * (T_RH / 100.0) * (g_star / 100.0) ** (1.0 / 6.0)
    Sf = (f / fp) ** 3 / (
        (1.0 + f / fp) ** (11.0 / 3.0) * (1.0 + 8.0 * math.pi * f / h_star)
    )
    return (
        3.35e-4
        * (1.0 / beta_H)
        * (kturb * alpha_val / (1.0 + alpha_val)) ** 1.5
        * (100.0 / g_star) ** (1.0 / 3.0)
        * v_w
        * Sf
    )


# ---------------------------------------------------------------------------
# Detector sensitivity curves  h^2 Omega_sens(f)
# ---------------------------------------------------------------------------
def _h2omega_from_Sh(f, Sh):
    H0 = 100.0 * H_PARAM * 1.0e3 / 3.086e22  # Hz
    return (2.0 * math.pi**2 / 3.0) * f**3 * Sh / H0**2 * H_PARAM**2


def sensitivity_LISA(f):
    f = np.asarray(f, dtype=float)
    L = 2.5e9
    f_star = 3.0e8 / (2.0 * math.pi * L)
    P_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / f) ** 4)
    P_acc = (3.0e-15) ** 2 * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8.0e-3) ** 4)
    Sc = (
        10.0
        / (3.0 * L**2)
        * (
            P_oms
            + 2.0 * (1.0 + np.cos(f / f_star) ** 2) * P_acc / (2.0 * math.pi * f) ** 4
        )
        * (1.0 + 0.6 * (f / f_star) ** 2)
    )
    return _h2omega_from_Sh(f, Sc)


def sensitivity_DECIGO(f):
    f = np.asarray(f, dtype=float)
    Sh = (
        7.05e-48 * (1.0 + (f / 7.36) ** 2)
        + 4.8e-51 * f ** (-4) / (1.0 + (f / 7.36) ** 2)
        + 5.33e-52 * f ** (-4)
    )
    return _h2omega_from_Sh(f, Sh)


def sensitivity_BBO(f):
    return sensitivity_DECIGO(f) / 100.0


def sensitivity_ET(f):
    f = np.asarray(f, dtype=float)
    x = f / 100.0
    Sh = (
        2.39e-27 * x ** (-15.64)
        + 0.349e-27 * x ** (-2.145)
        + 1.76e-27 * (1.0 + 0.12 * (x ** (-3.01)))
    ) ** 2
    return _h2omega_from_Sh(f, Sh)


def sensitivity_aLIGO(f):
    f = np.asarray(f, dtype=float)
    f0 = 215.0
    x = f / f0
    Sh = 1.0e-49 * (
        x ** (-4.14)
        - 5.0 * x ** (-2)
        + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2)
    )
    Sh = np.abs(Sh)
    return _h2omega_from_Sh(f, Sh)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_gw_spectrum(
    freq,
    omega_sw,
    omega_sw_supp,
    omega_turb,
    omega_env,
    alpha_val,
    beta_H,
    HR_star,
    T_n,
    T_RH,
    Uf,
    Upsilon,
    v_w,
    g_star,
    output_path,
    model_params=None,
):
    """Plot h^2 Omega_GW(f) with uncertainty band and detector curves."""
    omega_total = omega_sw + omega_turb + omega_env
    omega_total_supp = omega_sw_supp + omega_turb + omega_env

    fig, ax = plt.subplots(1, 1, figsize=(11, 7.5))

    # Uncertainty band for total signal
    ax.fill_between(
        freq, omega_total_supp, omega_total, color="royalblue", alpha=0.25, zorder=3
    )
    ax.loglog(freq, omega_total, color="navy", lw=2.5, zorder=5)
    ax.loglog(freq, omega_total_supp, color="navy", lw=1.0, ls=":", alpha=0.6, zorder=5)

    ax.loglog(freq, omega_turb, "r--", lw=1.2, alpha=0.6, zorder=4)
    if np.any(omega_env > 0):
        ax.loglog(freq, omega_env, "g--", lw=1.2, alpha=0.6, zorder=4)

    # Detector sensitivity curves
    f_lisa = np.logspace(-5, -0.5, 500)
    f_decigo = np.logspace(-3, 2, 500)
    f_bbo = np.logspace(-3, 2, 500)
    f_et = np.logspace(0.3, 3.5, 500)
    f_ligo = np.logspace(0.7, 3.7, 500)

    det_curves = [
        (f_lisa, sensitivity_LISA(f_lisa), "purple", "LISA"),
        (f_decigo, sensitivity_DECIGO(f_decigo), "orange", "DECIGO"),
        (f_bbo, sensitivity_BBO(f_bbo), "cyan", "BBO"),
        (f_et, sensitivity_ET(f_et), "brown", "ET"),
        (f_ligo, sensitivity_aLIGO(f_ligo), "green", "aLIGO"),
    ]
    for fv, sv, color, name in det_curves:
        ax.loglog(fv, sv, color=color, lw=1.5, alpha=0.55)

    # --- Annotations ---
    peak_idx = np.argmax(omega_total)
    f_pk = freq[peak_idx]
    o_pk = omega_total[peak_idx]
    ax.annotate(
        "TI PT GW",
        xy=(f_pk, o_pk),
        xytext=(f_pk * 0.04, o_pk * 10),
        fontsize=13,
        fontweight="bold",
        color="navy",
        arrowprops=dict(arrowstyle="->", color="navy", lw=1.2),
    )

    turb_peak_idx = np.argmax(omega_turb)
    f_tl = freq[max(turb_peak_idx - 400, 0)]
    o_tl = omega_turb[max(turb_peak_idx - 400, 0)]
    if o_tl > 1e-25:
        ax.text(
            f_tl,
            o_tl * 3,
            "Turbulence",
            fontsize=9,
            color="red",
            alpha=0.8,
            rotation=30,
        )

    _annotate_detector(ax, f_lisa, sensitivity_LISA(f_lisa), "LISA", "purple", 3e-3)
    _annotate_detector(
        ax, f_decigo, sensitivity_DECIGO(f_decigo), "DECIGO", "orange", 0.2
    )
    _annotate_detector(ax, f_bbo, sensitivity_BBO(f_bbo), "BBO", "cyan", 0.05)
    _annotate_detector(ax, f_et, sensitivity_ET(f_et), "ET", "brown", 5.0)
    _annotate_detector(ax, f_ligo, sensitivity_aLIGO(f_ligo), "aLIGO", "green", 50.0)

    ax.set_xlabel(r"$f$ [Hz]", fontsize=14)
    ax.set_ylabel(r"$h^2 \Omega_{\mathrm{GW}}$", fontsize=14)

    if T_RH >= 1e6:
        trh_str = rf"$T_{{\rm RH}} = {T_RH / 1e6:.2f}$ PeV"
    elif T_RH >= 1e3:
        trh_str = rf"$T_{{\rm RH}} = {T_RH / 1e3:.1f}$ TeV"
    else:
        trh_str = rf"$T_{{\rm RH}} = {T_RH:.0f}$ GeV"

    param_text = (
        rf"$T_n = {T_n:.0f}$ GeV"
        + "\n"
        + trh_str
        + "\n"
        + rf"$\beta/H = {beta_H:.0f}$"
        + "\n"
        + rf"$HR_* = {HR_star:.2e}$"
        + "\n"
        + rf"$\alpha = {alpha_val:.2e}$"
        + "\n"
        + rf"$\Upsilon = {Upsilon:.2e}$"
    )
    if model_params:
        param_text += (
            "\n"
            + rf"$M_S = {model_params['Ms']:.0f}$ GeV"
            + rf", $g_X = {model_params['gX']}$"
            + "\n"
            + rf"$\mu_* = {model_params['mu_star']:.0f}$ GeV"
        )
    ax.text(
        0.02,
        0.97,
        param_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
    )

    ax.set_xlim(1e-5, 1e5)
    ax.set_ylim(1e-20, 1e-5)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("GW Spectrum from Thermal Inflation Phase Transition", fontsize=13)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"  Saved figure: {output_path}")
    plt.close(fig)


def _annotate_detector(ax, fv, sv, name, color, f_target):
    idx = np.argmin(np.abs(fv - f_target))
    y_val = sv[idx]
    if y_val < 1e-20 or y_val > 1e-5:
        idx = np.argmin(sv)
        y_val = sv[idx]
    ax.annotate(
        name,
        xy=(fv[idx], y_val),
        xytext=(fv[idx], y_val * 0.15),
        fontsize=10,
        fontweight="bold",
        color=color,
        alpha=0.8,
        ha="center",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute GW spectrum from tunneling CSV data."
    )
    parser.add_argument("csv_path", help="Path to CSV with T and S3/T columns")
    parser.add_argument(
        "--vw", type=float, default=1.0, help="Bubble wall velocity (default: 1.0)"
    )
    parser.add_argument(
        "--g_star",
        type=float,
        default=G_STAR_DEFAULT,
        help="Relativistic DOF (default: 106.75)",
    )
    parser.add_argument(
        "--delV",
        type=float,
        default=DEL_V_DEFAULT,
        help="Vacuum energy difference in GeV^4 (default: 1e28)",
    )
    parser.add_argument(
        "--T_RH",
        type=float,
        default=None,
        help="Reheating temperature in GeV (overrides delV-based computation)",
    )
    parser.add_argument(
        "--Ms",
        type=float,
        default=None,
        help="Soft SUSY breaking scale M_S in GeV (used with --gX, --mu_star)",
    )
    parser.add_argument(
        "--gX",
        type=float,
        default=None,
        help="Coupling g_X (used with --Ms, --mu_star)",
    )
    parser.add_argument(
        "--mu_star",
        type=float,
        default=None,
        help="Renormalization scale mu_* in GeV (used with --Ms, --gX)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output figure path")
    parser.add_argument(
        "--csv_output", type=str, default=None, help="Output CSV path for GW parameters"
    )

    args = parser.parse_args()

    # Resolve model parameters if given
    has_model_params = (
        args.Ms is not None and args.gX is not None and args.mu_star is not None
    )
    if has_model_params:
        delV = compute_delV_model(args.Ms, args.gX, args.mu_star)
    else:
        delV = args.delV

    print("=" * 70)
    print("GW Spectrum from Thermal Inflation Phase Transition")
    print("  Sound waves: Hindmarsh et al. (2017), arXiv:2412.15864 eq.(4.31)")
    print("  Suppression: Ellis, Lewicki, No (2019)")
    print("=" * 70)
    print(f"  Input CSV:  {args.csv_path}")
    print(f"  v_w = {args.vw},  g* = {args.g_star},  delV = {delV:.2e} GeV^4")
    if has_model_params:
        print(
            f"  Model: M_S = {args.Ms:.1f} GeV,  g_X = {args.gX},  "
            f"mu_* = {args.mu_star:.1f} GeV"
        )
    print()

    T, S3_T = load_tunneling_data(args.csv_path)
    print(f"  Loaded {len(T)} data points")
    print(f"  T range: [{T.min():.1f}, {T.max():.1f}] GeV")
    print(f"  S3/T range: [{S3_T.min():.2f}, {S3_T.max():.2f}]")
    print()

    # Reheating temperature (priority: --T_RH > model params > delV)
    if args.T_RH is not None:
        T_RH = args.T_RH
        trh_source = "user-specified"
    elif has_model_params:
        T_RH = compute_T_RH_model(args.Ms, args.gX, args.mu_star, args.g_star)
        trh_source = "model params eq.(4.9)"
    else:
        T_RH = compute_T_RH(delV, args.g_star)
        trh_source = "delV"
    print("--- Reheating temperature ---")
    print(f"  T_RH = {T_RH:.4e} GeV  ({T_RH / 1e6:.4f} PeV = {T_RH / 1e3:.2f} TeV)")
    print(f"  source: {trh_source}")
    print()

    # Nucleation temperature
    print("--- Nucleation ---")
    T_n = find_nucleation_temp(T, S3_T, delV)
    print(f"  T_n = {T_n:.2f} GeV  ({T_n / 1e3:.4f} TeV)")
    print(f"  T_RH / T_n = {T_RH / T_n:.1f}")

    # beta/H
    beta_H = compute_beta_over_H(T, S3_T, T_n)
    print(f"  beta/H = {beta_H:.2f}  (-d ln Gamma / d ln T)")

    cs_s = CubicSpline(T, S3_T)
    dS3T_dT_at_Tn = float(cs_s(T_n, 1))
    beta_H_approx = T_n * dS3T_dT_at_Tn
    print(f"  beta/H = {beta_H_approx:.2f}  (T d(S3/T)/dT)")

    # HR_*
    HR_star = compute_HR_star(beta_H)
    print(f"  HR_* = (8pi)^{{1/3}} / (beta/H) = {HR_star:.4e}")

    # alpha
    alpha_val = compute_alpha(T_n, delV, args.g_star)
    print(f"  alpha = {alpha_val:.4e}")
    print()

    # Efficiency factors
    kv = kappa_v(alpha_val)
    kt = kappa_turb_frac() * kv
    kphi = 0.0
    if alpha_val > 10:
        kphi = max(0.0, 1.0 - kv)

    # RMS fluid velocity
    Uf = compute_Uf(kv, alpha_val)

    # Suppression factor (Ellis, Lewicki, No 2019)
    Upsilon = min(1.0, HR_star / Uf)

    print("--- Efficiency & fluid ---")
    print(f"  kappa_v    = {kv:.4f}")
    print(f"  kappa_turb = {kt:.4f}")
    print(f"  kappa_phi  = {kphi:.4f}")
    print(f"  Uf (RMS)   = {Uf:.4f}")
    print(f"  Upsilon    = {Upsilon:.4e}  (suppression factor)")
    print()

    # Peak frequencies
    fp_sw = f_peak_sw(HR_star, T_RH, args.g_star)
    fp_env = f_peak_env(beta_H, T_RH, args.g_star, args.vw)
    fp_turb = f_peak_turb(beta_H, T_RH, args.g_star, args.vw)

    print("--- Peak frequencies (T_RH redshift) ---")
    print(f"  f_peak (sound wave)  = {fp_sw:.4e} Hz")
    print(f"  f_peak (turbulence)  = {fp_turb:.4e} Hz")
    print(f"  f_peak (envelope)    = {fp_env:.4e} Hz")
    print()

    # Compute spectra
    freq = np.logspace(-5, 5, 3000)

    omega_sw = gw_sound_wave(freq, HR_star, Uf, args.g_star, fp_sw)
    omega_sw_supp = omega_sw * Upsilon

    omega_env = gw_envelope(freq, alpha_val, beta_H, args.g_star, kphi, fp_env)
    omega_turb = gw_turbulence(
        freq, alpha_val, beta_H, T_RH, args.g_star, args.vw, kt, fp_turb
    )

    omega_total = omega_sw + omega_turb + omega_env
    omega_total_supp = omega_sw_supp + omega_turb + omega_env

    peak_total = np.max(omega_total)
    f_at_peak = freq[np.argmax(omega_total)]
    peak_supp = np.max(omega_total_supp)
    f_at_peak_supp = freq[np.argmax(omega_total_supp)]

    print("--- Peak amplitude ---")
    print(f"  h^2 Omega (nominal)    = {peak_total:.4e}  at  f = {f_at_peak:.4e} Hz")
    print(
        f"  h^2 Omega (suppressed) = {peak_supp:.4e}  at  f = {f_at_peak_supp:.4e} Hz"
    )
    print()

    # Output figure
    if args.output:
        fig_path = args.output
    else:
        csv_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        fig_path = f"figs/gw_spectrum/gw_spectrum_{csv_name}.png"

    model_params = None
    if has_model_params:
        model_params = {"Ms": args.Ms, "gX": args.gX, "mu_star": args.mu_star}

    plot_gw_spectrum(
        freq,
        omega_sw,
        omega_sw_supp,
        omega_turb,
        omega_env,
        alpha_val,
        beta_H,
        HR_star,
        T_n,
        T_RH,
        Uf,
        Upsilon,
        args.vw,
        args.g_star,
        fig_path,
        model_params,
    )

    # Save parameters CSV
    if args.csv_output:
        csv_out = args.csv_output
    else:
        csv_dir = os.path.dirname(args.csv_path)
        csv_out = os.path.join(csv_dir, "gw_parameters.csv")

    params_df = pd.DataFrame(
        [
            {
                "T_n_GeV": T_n,
                "T_RH_GeV": T_RH,
                "beta_over_H_general": beta_H,
                "beta_over_H_approx": beta_H_approx,
                "HR_star": HR_star,
                "alpha": alpha_val,
                "Uf": Uf,
                "Upsilon": Upsilon,
                "v_w": args.vw,
                "g_star": args.g_star,
                "delV_GeV4": delV,
                "kappa_v": kv,
                "kappa_turb": kt,
                "kappa_phi": kphi,
                "f_peak_sw_Hz": fp_sw,
                "f_peak_turb_Hz": fp_turb,
                "f_peak_env_Hz": fp_env,
                "h2_Omega_peak_nominal": peak_total,
                "h2_Omega_peak_suppressed": peak_supp,
                "f_at_peak_Hz": f_at_peak,
            }
        ]
    )
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    params_df.to_csv(csv_out, index=False)
    print(f"  Saved parameters: {csv_out}")

    print()
    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
