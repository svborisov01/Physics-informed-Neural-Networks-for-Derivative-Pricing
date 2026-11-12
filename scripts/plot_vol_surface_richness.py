#!/usr/bin/env python3
"""
Illustrative comparison of volatility-surface richness:

  Black–Scholes → Heston → 1-factor Bergomi → 2-factor Bergomi

These are stylized implied-vol surfaces (not PINN outputs) that show what
each model *can* generate: flat / one-factor skew / lognormal skew /
two-timescale term structure of skew.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bs_iv(k, tau, sigma=0.20):
    return np.full_like(k, sigma, dtype=float)


def heston_like_iv(k, tau, v0=0.04, rho=-0.7, volvol=0.5, kappa=2.0):
    """Sticky skew that weakens mildly with maturity (single timescale)."""
    atm = np.sqrt(v0) * (1.0 + 0.05 * np.exp(-kappa * tau))
    skew = rho * volvol * 0.35 * np.exp(-0.6 * tau) * k
    smile = 0.08 * (k ** 2) * np.exp(-0.3 * tau)
    return np.clip(atm + skew + smile, 0.05, 0.8)


def bergomi1_like_iv(k, tau, xi0=0.04, omega=1.0, rho=-0.5, kappa=2.0):
    """Lognormal-vol skew; stronger short-dated wings than Heston."""
    atm = np.sqrt(xi0) * (1.0 + 0.08 * omega * np.exp(-kappa * tau))
    skew = rho * omega * 0.28 * np.exp(-kappa * 0.5 * tau) * k
    # Stronger short-term smile (rough-ish / lognormal effect, stylized)
    smile = 0.14 * omega * (k ** 2) * np.exp(-1.2 * tau)
    return np.clip(atm + skew + smile, 0.05, 0.9)


def bergomi2_like_iv(
    k,
    tau,
    xi0=0.04,
    omega1=1.0,
    omega2=0.6,
    kappa1=5.0,
    kappa2=0.6,
    rho1=-0.7,
    rho2=-0.3,
):
    """
    Two timescales: fast factor dominates short skew/smile;
    slow factor keeps longer-dated skew alive.
    """
    w_fast = np.exp(-kappa1 * tau)
    w_slow = np.exp(-kappa2 * tau)
    atm = np.sqrt(xi0) * (
        1.0 + 0.10 * omega1 * w_fast + 0.12 * omega2 * w_slow
    )
    skew = (
        rho1 * omega1 * 0.30 * w_fast + rho2 * omega2 * 0.22 * w_slow
    ) * k
    smile = (
        0.16 * omega1 * (k ** 2) * w_fast + 0.06 * omega2 * (k ** 2) * w_slow
    )
    return np.clip(atm + skew + smile, 0.05, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="bergomi2/artifacts")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    k = np.linspace(-0.4, 0.4, 81)  # log-moneyness
    taus = np.array([0.25, 0.5, 1.0, 2.0])
    models = [
        ("Black–Scholes\n(constant σ)", bs_iv, "C0"),
        ("Heston\n(1 variance factor)", heston_like_iv, "C1"),
        ("1-factor Bergomi\n(1 OU factor)", bergomi1_like_iv, "C2"),
        ("2-factor Bergomi\n(fast + slow OU)", bergomi2_like_iv, "C3"),
    ]

    # --- Smile slices at several maturities ---
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), dpi=150, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (name, fn, color) in zip(axes, models):
        for tau in taus:
            iv = fn(k, tau)
            ax.plot(k, iv, label=rf"$\tau={tau:g}$")
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

    # --- Term structure of ATM vol and 25Δ risk-reversal proxy ---
    tau_fine = np.linspace(0.05, 2.0, 80)
    k_rr = 0.15  # proxy wing for RR ≈ IV(-k) - IV(+k)

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

    # --- 3D surface for 2F vs BS ---
    Kgrid = k
    Tgrid = np.linspace(0.1, 2.0, 40)
    KK, TT = np.meshgrid(Kgrid, Tgrid, indexing="ij")
    iv_bs = bs_iv(KK, TT)
    iv_2f = bergomi2_like_iv(KK, TT)

    fig = plt.figure(figsize=(12.0, 5.0), dpi=150)
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for ax, Z, title, cmap in (
        (ax1, iv_bs, "Black–Scholes (flat)", "viridis"),
        (ax2, iv_2f, "2-factor Bergomi (rich)", "turbo"),
    ):
        surf = ax.plot_surface(
            KK, TT, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95
        )
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$\tau$")
        ax.set_zlabel(r"$\sigma_{\mathrm{IV}}$")
        ax.set_title(title)
        ax.set_zlim(0.05, 0.55)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    fig.suptitle("Implied-vol surface: constant vol vs two-factor Bergomi", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "vol_surface_bs_vs_bergomi2.png", bbox_inches="tight")
    plt.close(fig)

    print("Wrote plots to", out)


if __name__ == "__main__":
    main()
