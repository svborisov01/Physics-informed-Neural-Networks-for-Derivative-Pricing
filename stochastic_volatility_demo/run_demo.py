#!/usr/bin/env python3
"""
Stochastic-volatility demo:

  1. Download (or synthesize) listed option IVs
  2. Fit BS / Heston / 1F Bergomi / 2F Bergomi parametric surfaces
  3. Plot fits across *moneyness* (smiles) and across *tenors* (term structure)

The headline result is that **two-factor Bergomi** achieves the lowest error
on short *and* long tenors simultaneously.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from stochastic_volatility_demo.fetch_market import load_or_fetch
from stochastic_volatility_demo.fit_surfaces import MODEL_FN, fit_all

ART = ROOT / "artifacts"

COLORS = {
    "Black–Scholes": "0.35",
    "Heston-like": "C1",
    "1-factor Bergomi": "C2",
    "2-factor Bergomi": "C3",
}


def _select_tenors(taus: np.ndarray, n: int = 4) -> list[float]:
    uniq = np.array(sorted(set(np.round(taus, 6))))
    if len(uniq) <= n:
        return [float(t) for t in uniq]
    idx = np.unique(np.linspace(0, len(uniq) - 1, n).astype(int))
    return [float(uniq[i]) for i in idx]


def plot_market_surface(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    sc = ax.scatter(
        df["k"],
        df["tau"],
        c=df["iv"],
        cmap="turbo",
        s=28,
        alpha=0.9,
        edgecolors="none",
    )
    ax.set_xlabel(r"Log-moneyness $k=\log(K/S)$")
    ax.set_ylabel(r"Maturity $\tau$")
    ax.set_title("Market implied volatility surface (OTM strip)")
    fig.colorbar(sc, ax=ax, label=r"$\sigma_{\mathrm{IV}}$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_smile_fits(df: pd.DataFrame, fits, path: Path):
    tenors = _select_tenors(df["tau"].to_numpy(), n=4)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), dpi=150, sharey=True)
    axes = axes.ravel()
    k_line = np.linspace(df["k"].min(), df["k"].max(), 120)
    for ax, tau in zip(axes, tenors):
        mkt = df[np.isclose(df["tau"], tau, atol=1e-3)].sort_values("k")
        ax.scatter(mkt["k"], mkt["iv"], s=22, color="k", zorder=5, label="Market")
        for fr in fits:
            pred = MODEL_FN[fr.name](k_line, np.full_like(k_line, tau), **fr.params)
            ax.plot(
                k_line,
                pred,
                color=COLORS[fr.name],
                lw=2.0 if "2-factor" in fr.name else 1.5,
                label=fr.name,
            )
        ax.set_title(rf"$\tau={tau:.2f}$ y")
        ax.set_xlabel(r"$k$")
        ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Fitted smiles vs market — two clocks needed for short+long", y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _market_term_structure(df: pd.DataFrame, k_star: float = 0.0, k_tol: float = 0.03):
    """Nearest-to-target-k IV at each tenor (for ATM / wing term structures)."""
    rows = []
    for tau, g in df.groupby(df["tau"].round(6)):
        j = (g["k"] - k_star).abs().idxmin()
        if abs(float(g.loc[j, "k"]) - k_star) > k_tol:
            continue
        rows.append((float(tau), float(g.loc[j, "iv"]), float(g.loc[j, "k"])))
    rows.sort(key=lambda z: z[0])
    if not rows:
        return np.array([]), np.array([])
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def _market_skew_term_structure(df: pd.DataFrame, k_rr: float = 0.10):
    """Skew proxy IV(-k_rr) − IV(+k_rr) across tenors."""
    rows = []
    for tau, g in df.groupby(df["tau"].round(6)):
        left = g.iloc[(g["k"] + k_rr).abs().argmin()]
        right = g.iloc[(g["k"] - k_rr).abs().argmin()]
        if abs(float(left["k"]) + k_rr) > 0.04 or abs(float(right["k"]) - k_rr) > 0.04:
            continue
        rows.append((float(tau), float(left["iv"]) - float(right["iv"])))
    rows.sort(key=lambda z: z[0])
    if not rows:
        return np.array([]), np.array([])
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def plot_tenor_fits(df: pd.DataFrame, fits, path: Path):
    """
    Fit quality across maturity: ATM term structure, wing term structure,
    and skew term structure — the tenor analogue of the moneyness smile plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=150)
    tau_line = np.linspace(float(df["tau"].min()), float(df["tau"].max()), 80)

    # ATM
    t_m, iv_m = _market_term_structure(df, k_star=0.0)
    axes[0].scatter(t_m, iv_m, s=28, color="k", zorder=5, label="Market")
    for fr in fits:
        pred = MODEL_FN[fr.name](
            np.zeros_like(tau_line), tau_line, **fr.params
        )
        axes[0].plot(
            tau_line,
            pred,
            color=COLORS[fr.name],
            lw=2.2 if "2-factor" in fr.name else 1.5,
            label=fr.name,
        )
    axes[0].set_title("ATM IV term structure")
    axes[0].set_xlabel(r"Maturity $\tau$")
    axes[0].set_ylabel(r"$\sigma_{\mathrm{ATM}}(\tau)$")

    # 10-delta-ish put wing (k = -0.10)
    t_w, iv_w = _market_term_structure(df, k_star=-0.10, k_tol=0.04)
    axes[1].scatter(t_w, iv_w, s=28, color="k", zorder=5, label="Market")
    for fr in fits:
        pred = MODEL_FN[fr.name](
            np.full_like(tau_line, -0.10), tau_line, **fr.params
        )
        axes[1].plot(
            tau_line,
            pred,
            color=COLORS[fr.name],
            lw=2.2 if "2-factor" in fr.name else 1.5,
            label=fr.name,
        )
    axes[1].set_title(r"Put wing $k=-0.10$ vs $\tau$")
    axes[1].set_xlabel(r"Maturity $\tau$")
    axes[1].set_ylabel("IV")

    # Skew
    t_s, sk_m = _market_skew_term_structure(df, k_rr=0.10)
    axes[2].scatter(t_s, sk_m, s=28, color="k", zorder=5, label="Market")
    for fr in fits:
        left = MODEL_FN[fr.name](np.full_like(tau_line, -0.10), tau_line, **fr.params)
        right = MODEL_FN[fr.name](np.full_like(tau_line, 0.10), tau_line, **fr.params)
        axes[2].plot(
            tau_line,
            left - right,
            color=COLORS[fr.name],
            lw=2.2 if "2-factor" in fr.name else 1.5,
            label=fr.name,
        )
    axes[2].set_title(r"Skew $\mathrm{IV}(-0.10)-\mathrm{IV}(+0.10)$")
    axes[2].set_xlabel(r"Maturity $\tau$")
    axes[2].set_ylabel("Skew")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Fitted term structures vs market — moneyness held fixed, vary maturity",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_errors_by_tenor(fits, path: Path):
    tenors = sorted({t for fr in fits for t in fr.rmse_by_tenor})
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    for fr in fits:
        ys = [fr.rmse_by_tenor.get(t, np.nan) for t in tenors]
        ax.plot(
            tenors,
            ys,
            "o-",
            color=COLORS[fr.name],
            lw=2.2 if "2-factor" in fr.name else 1.5,
            label=f"{fr.name} (RMSE={fr.rmse:.3f})",
        )
    ax.set_xlabel(r"Maturity $\tau$")
    ax.set_ylabel("IV RMSE")
    ax.set_title("Fit error by tenor: 2-factor Bergomi stays low short *and* long")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_errors_by_moneyness(fits, path: Path):
    """Companion to tenor RMSE: error buckets across log-moneyness."""
    # Use 2F keys as canonical bin order when present
    keys = None
    for fr in fits:
        if "2-factor" in fr.name:
            keys = list(fr.rmse_by_moneyness.keys())
            break
    if not keys:
        keys = list(fits[0].rmse_by_moneyness.keys())

    x = np.arange(len(keys))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=150)
    for i, fr in enumerate(fits):
        ys = [fr.rmse_by_moneyness.get(k, np.nan) for k in keys]
        ax.bar(
            x + (i - 1.5) * width,
            ys,
            width=width,
            color=COLORS[fr.name],
            label=fr.name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace(":", "–") for k in keys], rotation=15)
    ax.set_xlabel(r"Log-moneyness bucket $k$")
    ax.set_ylabel("IV RMSE")
    ax.set_title("Fit error by moneyness: wings vs ATM")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(symbol: str = "SPY"):
    ART.mkdir(parents=True, exist_ok=True)
    print(f"Fetching option surface for {symbol}…")
    df, meta = load_or_fetch(symbol=symbol)
    print(
        f"Using {meta.get('source')} | spot={meta.get('spot')} | "
        f"n={len(df)} quotes | tenors={len(meta.get('tenors', []))}"
    )

    k = df["k"].to_numpy(float)
    tau = df["tau"].to_numpy(float)
    iv = df["iv"].to_numpy(float)
    fits = fit_all(k, tau, iv)

    plot_market_surface(df, ART / "market_iv_surface.png")
    plot_smile_fits(df, fits, ART / "model_fits_smiles.png")
    plot_tenor_fits(df, fits, ART / "model_fits_tenors.png")
    plot_errors_by_tenor(fits, ART / "fit_errors_by_tenor.png")
    plot_errors_by_moneyness(fits, ART / "fit_errors_by_moneyness.png")

    report = {
        "market": meta,
        "models": [
            {
                "name": fr.name,
                "rmse": fr.rmse,
                "mae": fr.mae,
                "params": fr.params,
                "rmse_by_tenor": {str(a): b for a, b in fr.rmse_by_tenor.items()},
                "rmse_by_moneyness": {
                    str(a): b for a, b in fr.rmse_by_moneyness.items()
                },
            }
            for fr in fits
        ],
        "ranking": [fr.name for fr in sorted(fits, key=lambda z: z.rmse)],
    }
    (ART / "fit_report.json").write_text(json.dumps(report, indent=2))

    print("\nIV fit RMSE (lower is better):")
    for fr in sorted(fits, key=lambda z: z.rmse):
        print(f"  {fr.name:18s}  RMSE={fr.rmse:.4f}  MAE={fr.mae:.4f}")
    print(f"\nWrote artifacts to {ART}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    main(sym)
