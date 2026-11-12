#!/usr/bin/env python3
"""
Illustrative comparison of volatility-surface richness:

  Black–Scholes → Heston → 1-factor Bergomi → 2-factor Bergomi

These are stylized implied-vol surfaces (not PINN outputs) that show what
each model *can* generate: flat / one-factor skew / lognormal smile /
two-timescale term structure of smile+skew.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bs_iv(k, tau, sigma=0.20):
    return np.full_like(k, float(sigma), dtype=float)


def heston_like_iv(k, tau, v0=0.04, rho=-0.65, volvol=0.55, kappa=2.0):
    """Single-timescale skew + mild smile (CIR variance factor)."""
    atm = np.sqrt(v0) * (1.0 + 0.04 * np.exp(-kappa * tau))
    skew = rho * volvol * 0.32 * np.exp(-0.55 * tau) * k
    smile = 0.10 * (k ** 2) * np.exp(-0.35 * tau)
    return np.clip(atm + skew + smile, 0.05, 0.85)


def bergomi1_like_iv(k, tau, xi0=0.04, omega=1.2, rho=-0.35, kappa=2.0):
    """
    1-factor Bergomi: lognormal variance → stronger short-dated wings.
    Milder |ρ| so the U-shaped smile is visible alongside equity-style skew.
    """
    atm = np.sqrt(xi0) * (1.0 + 0.10 * omega * np.exp(-kappa * tau))
    skew = rho * omega * 0.22 * np.exp(-0.7 * kappa * tau) * k
    smile = 0.28 * omega * (k ** 2) * np.exp(-1.4 * tau)
    return np.clip(atm + skew + smile, 0.05, 0.95)


def bergomi2_like_iv(
    k,
    tau,
    xi0=0.04,
    omega1=1.4,
    omega2=0.7,
    kappa1=8.0,
    kappa2=0.5,
    rho1=-0.25,
    rho2=-0.15,
):
    """
    Two timescales with a *visible ATM smile trough*:
    fast factor → short-dated U-shaped wings; slow factor → longer skew/ATM.
    Milder correlations keep the smile from collapsing into pure skew.
    """
    w_fast = np.exp(-kappa1 * tau)
    w_slow = np.exp(-kappa2 * tau)
    atm = np.sqrt(xi0) * (1.0 + 0.12 * omega1 * w_fast + 0.10 * omega2 * w_slow)
    skew = (rho1 * omega1 * 0.18 * w_fast + rho2 * omega2 * 0.20 * w_slow) * k
    # Dominant quadratic wings — this is the "usual smile"
    smile = 0.38 * omega1 * (k ** 2) * w_fast + 0.12 * omega2 * (k ** 2) * w_slow
    return np.clip(atm + skew + smile, 0.05, 1.05)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="results/bergomi2")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    k = np.linspace(-0.45, 0.45, 91)
    taus = np.array([0.25, 0.5, 1.0, 2.0])
    models = [
        ("Black–Scholes\n(constant σ)", bs_iv, "C0"),
        ("Heston\n(1 variance factor)", heston_like_iv, "C1"),
        ("1-factor Bergomi\n(1 OU factor)", bergomi1_like_iv, "C2"),
        ("2-factor Bergomi\n(fast + slow OU)", bergomi2_like_iv, "C3"),
    ]

    # --- Smile slices ---
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), dpi=150, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (name, fn, _color) in zip(axes, models):
        for tau in taus:
            ax.plot(k, fn(k, tau), label=rf"$\tau={tau:g}$")
        ax.axvline(0.0, color="k", lw=0.6, alpha=0.25)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(r"Log-moneyness $k=\log(K/F)$")
        ax.set_ylabel("Implied volatility")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        "Volatility smile richness across models (stylized)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out / "vol_smile_richness.png", bbox_inches="tight")
    plt.close(fig)

    # --- Term structure of ATM vol and skew proxy ---
    tau_fine = np.linspace(0.05, 2.0, 80)
    k_rr = 0.15
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=150)
    for name, fn, color in models:
        atm = np.array([fn(np.array([0.0]), t)[0] for t in tau_fine])
        rr = np.array(
            [
                fn(np.array([-k_rr]), t)[0] - fn(np.array([k_rr]), t)[0]
                for t in tau_fine
            ]
        )
        label = name.split("\n")[0]
        axes[0].plot(tau_fine, atm, color=color, label=label, lw=2)
        axes[1].plot(tau_fine, rr, color=color, label=label, lw=2)
    axes[0].set_title("ATM implied vol term structure")
    axes[0].set_xlabel(r"$\tau$")
    axes[0].set_ylabel(r"$\sigma_{\mathrm{ATM}}(\tau)$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_title(r"Skew proxy $\mathrm{IV}(-0.15)-\mathrm{IV}(+0.15)$")
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel("Skew")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Term-structure flexibility: one vs two timescales", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "vol_termstructure_richness.png", bbox_inches="tight")
    plt.close(fig)

    # --- 3D surface: BS flat vs 2F Bergomi with a clear smile trough ---
    # Short-maturity window + strong quadratic wings so the U-trough is obvious
    # (equity-style mild skew must not erase the usual smile shape).
    Kgrid = np.linspace(-0.50, 0.50, 71)
    Tgrid = np.linspace(0.05, 1.0, 50)
    KK, TT = np.meshgrid(Kgrid, Tgrid, indexing="ij")
    iv_bs = bs_iv(KK, TT)
    iv_2f = bergomi2_like_iv(
        KK,
        TT,
        omega1=1.85,
        omega2=0.45,
        kappa1=12.0,
        kappa2=0.40,
        rho1=-0.15,
        rho2=-0.08,
    )

    fig = plt.figure(figsize=(12.4, 5.4), dpi=150)
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for ax, Z, title, cmap, zlim in (
        (ax1, iv_bs, "Black–Scholes (flat surface)", "viridis", (0.12, 0.42)),
        (
            ax2,
            iv_2f,
            "2-factor Bergomi (U-shaped smile)",
            "turbo",
            (0.12, 0.42),
        ),
    ):
        surf = ax.plot_surface(
            KK,
            TT,
            Z,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.98,
            rstride=1,
            cstride=1,
        )
        # Look along maturity so the moneyness U-trough reads clearly
        ax.view_init(elev=22, azim=-55)
        ax.set_xlabel(r"Log-moneyness $k$")
        ax.set_ylabel(r"Maturity $\tau$")
        ax.set_zlabel(r"$\sigma_{\mathrm{IV}}$")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_zlim(*zlim)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08, label=r"$\sigma_{\mathrm{IV}}$")
    fig.suptitle(
        "Implied-vol surface: flat Black–Scholes vs two-factor Bergomi smile",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "vol_surface_bs_vs_bergomi2.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Extra: short-maturity smile slice highlighting the U-shape
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for tau, ls in ((0.1, "-"), (0.25, "-"), (0.5, "--"), (1.0, ":")):
        ax.plot(
            k,
            bergomi2_like_iv(
                k, tau, omega1=1.85, omega2=0.45, kappa1=12.0, kappa2=0.40, rho1=-0.15, rho2=-0.08
            ),
            ls=ls,
            label=rf"2F Bergomi $\tau={tau:g}$",
        )
    ax.plot(k, bs_iv(k, 0.25), color="0.35", lw=2, label="Black–Scholes")
    ax.axvline(0.0, color="k", lw=0.6, alpha=0.3)
    ax.set_xlabel(r"Log-moneyness $k=\log(K/F)$")
    ax.set_ylabel("Implied volatility")
    ax.set_title("Short-dated smile: 2-factor Bergomi vs flat BS")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "vol_smile_bergomi2_short.png", bbox_inches="tight")
    plt.close(fig)

    print("Wrote plots to", out)


if __name__ == "__main__":
    main()
