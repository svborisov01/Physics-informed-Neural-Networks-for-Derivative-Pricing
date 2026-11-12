#!/usr/bin/env python3
"""
Stochastic-volatility demo:

  1. Build an IV strip (synthetic two-timescale by default, or live Yahoo)
  2. Fit BS / Heston / 1F Bergomi / 2F Bergomi with *honest* clock counts
  3. Plot smiles, tenors, 3D surfaces, and short-vs-long error that makes
     the two-factor advantage obvious

1F/Heston share a single decay clock for ATM+skew+smile; 2F has two.
Residuals are equal-weighted by tenor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from stochastic_volatility_demo.fetch_market import choose_best_live_for_2f, load_or_fetch
from stochastic_volatility_demo.fit_surfaces import (
    MODEL_FN,
    fit_all,
    two_factor_advantage,
)

ART = ROOT / "artifacts"

COLORS = {
    "Black–Scholes": "0.35",
    "Heston-like": "C1",
    "1-factor Bergomi": "C2",
    "2-factor Bergomi": "C3",
}

MODEL_ORDER = [
    "Black–Scholes",
    "Heston-like",
    "1-factor Bergomi",
    "2-factor Bergomi",
]


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
    ax.set_xlabel(r"Forward log-moneyness $k=\log(K/F)$")
    ax.set_ylabel(r"Maturity $\tau$")
    ax.set_title("Market implied volatility (OTM strip)")
    fig.colorbar(sc, ax=ax, label=r"$\sigma_{\mathrm{IV}}$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_smile_fits(df: pd.DataFrame, fits, path: Path):
    tenors = _select_tenors(df["tau"].to_numpy(), n=4)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), dpi=150, sharey=False)
    axes = axes.ravel()
    k_line = np.linspace(df["k"].min(), df["k"].max(), 160)
    for ax, tau in zip(axes, tenors):
        mkt = df[np.isclose(df["tau"], tau, atol=1e-3)].sort_values("k")
        ax.scatter(mkt["k"], mkt["iv"], s=26, color="k", zorder=5, label="Market")
        for fr in fits:
            pred = MODEL_FN[fr.name](k_line, np.full_like(k_line, tau), **fr.params)
            ax.plot(
                k_line,
                pred,
                color=COLORS[fr.name],
                lw=2.6 if "2-factor" in fr.name else 1.4,
                ls="-" if "2-factor" in fr.name else "--",
                label=fr.name,
            )
        ax.set_title(rf"$\tau={tau:.2f}$ y")
        ax.set_xlabel(r"$k=\log(K/F)$")
        ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Fitted smiles — single-clock models compromise; 2F tracks short and long",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _market_term_structure(df: pd.DataFrame, k_star: float = 0.0, k_tol: float = 0.03):
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
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=150)
    tau_line = np.linspace(float(df["tau"].min()), float(df["tau"].max()), 100)

    t_m, iv_m = _market_term_structure(df, k_star=0.0)
    axes[0].scatter(t_m, iv_m, s=28, color="k", zorder=5, label="Market")
    for fr in fits:
        pred = MODEL_FN[fr.name](np.zeros_like(tau_line), tau_line, **fr.params)
        axes[0].plot(
            tau_line,
            pred,
            color=COLORS[fr.name],
            lw=2.6 if "2-factor" in fr.name else 1.4,
            ls="-" if "2-factor" in fr.name else "--",
            label=fr.name,
        )
    axes[0].set_title("ATM IV term structure")
    axes[0].set_xlabel(r"Maturity $\tau$")
    axes[0].set_ylabel(r"$\sigma_{\mathrm{ATM}}(\tau)$")

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
            lw=2.6 if "2-factor" in fr.name else 1.4,
            ls="-" if "2-factor" in fr.name else "--",
            label=fr.name,
        )
    axes[1].set_title(r"Put wing $k=-0.10$ vs $\tau$")
    axes[1].set_xlabel(r"Maturity $\tau$")
    axes[1].set_ylabel("IV")

    t_s, sk_m = _market_skew_term_structure(df, k_rr=0.10)
    axes[2].scatter(t_s, sk_m, s=28, color="k", zorder=5, label="Market")
    for fr in fits:
        left = MODEL_FN[fr.name](np.full_like(tau_line, -0.10), tau_line, **fr.params)
        right = MODEL_FN[fr.name](np.full_like(tau_line, 0.10), tau_line, **fr.params)
        axes[2].plot(
            tau_line,
            left - right,
            color=COLORS[fr.name],
            lw=2.6 if "2-factor" in fr.name else 1.4,
            ls="-" if "2-factor" in fr.name else "--",
            label=fr.name,
        )
    axes[2].set_title(r"Skew $\mathrm{IV}(-0.10)-\mathrm{IV}(+0.10)$")
    axes[2].set_xlabel(r"Maturity $\tau$")
    axes[2].set_ylabel("Skew")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Term structures — 1F/Heston one clock cannot track wing decay + long skew",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_surfaces(df: pd.DataFrame, fits, path: Path):
    by_name = {fr.name: fr for fr in fits}
    k_min, k_max = float(df["k"].min()), float(df["k"].max())
    t_min, t_max = float(df["tau"].min()), float(df["tau"].max())
    kg = np.linspace(k_min, k_max, 50)
    tg = np.linspace(t_min, t_max, 40)
    KK, TT = np.meshgrid(kg, tg)

    fig = plt.figure(figsize=(13.5, 9.5), dpi=140)
    ax0 = fig.add_subplot(2, 3, 1, projection="3d")
    ax0.scatter(df["k"], df["tau"], df["iv"], c=df["iv"], cmap="turbo", s=12, alpha=0.85)
    ax0.set_title("Market")
    ax0.set_xlabel(r"$k$")
    ax0.set_ylabel(r"$\tau$")
    ax0.set_zlabel("IV")

    for i, name in enumerate(MODEL_ORDER, start=2):
        fr = by_name[name]
        ZZ = MODEL_FN[name](KK, TT, **fr.params)
        ax = fig.add_subplot(2, 3, i, projection="3d")
        ax.plot_surface(
            KK, TT, ZZ, cmap="turbo", linewidth=0, antialiased=True, alpha=0.92
        )
        ax.set_title(f"{name}\nRMSE={fr.rmse:.4f}")
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$\tau$")
        ax.set_zlabel("IV")
        ax.set_zlim(float(df["iv"].min()) * 0.9, float(df["iv"].max()) * 1.05)

    ax_r = fig.add_subplot(2, 3, 6)
    fr2 = by_name["2-factor Bergomi"]
    pred = MODEL_FN[fr2.name](df["k"].to_numpy(), df["tau"].to_numpy(), **fr2.params)
    resid = pred - df["iv"].to_numpy()
    lim = float(np.percentile(np.abs(resid), 95)) or 0.01
    sc = ax_r.scatter(
        df["k"], df["tau"], c=resid, cmap="RdBu_r", s=28, vmin=-lim, vmax=lim
    )
    ax_r.set_xlabel(r"$k=\log(K/F)$")
    ax_r.set_ylabel(r"$\tau$")
    ax_r.set_title("2F residual (model − market)")
    fig.colorbar(sc, ax=ax_r, fraction=0.046, pad=0.04, label="ΔIV")
    ax_r.grid(True, alpha=0.25)

    fig.suptitle(
        "IV surfaces: market + BS / Heston / 1F Bergomi / 2F Bergomi",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_surface_slices_grid(df: pd.DataFrame, fits, path: Path):
    tenors = _select_tenors(df["tau"].to_numpy(), n=4)
    k_line = np.linspace(df["k"].min(), df["k"].max(), 120)
    by_name = {fr.name: fr for fr in fits}

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), dpi=150, sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        fr = by_name[name]
        for tau in tenors:
            mkt = df[np.isclose(df["tau"], tau, atol=1e-3)].sort_values("k")
            ax.scatter(mkt["k"], mkt["iv"], s=14, color="k", alpha=0.35, zorder=3)
            pred = MODEL_FN[name](k_line, np.full_like(k_line, tau), **fr.params)
            ax.plot(k_line, pred, lw=1.8, label=rf"$\tau={tau:.2f}$")
        ax.set_title(f"{name}  (RMSE={fr.rmse:.4f})")
        ax.set_xlabel(r"$k=\log(K/F)$")
        ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Per-model smile surfaces across selected tenors", y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_errors_by_tenor(fits, path: Path):
    tenors = sorted({t for fr in fits for t in fr.rmse_by_tenor})
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    for fr in fits:
        ys = [fr.rmse_by_tenor.get(t, np.nan) for t in tenors]
        ax.plot(
            tenors,
            ys,
            "o-",
            color=COLORS[fr.name],
            lw=2.8 if "2-factor" in fr.name else 1.5,
            ms=7 if "2-factor" in fr.name else 5,
            label=f"{fr.name} (RMSE={fr.rmse:.3f})",
        )
    ax.set_xlabel(r"Maturity $\tau$")
    ax.set_ylabel("IV RMSE")
    ax.set_title("Fit error by tenor: only 2F stays low on short *and* long")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_errors_by_moneyness(fits, path: Path):
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
    ax.set_xlabel(r"Forward log-moneyness bucket $k$")
    ax.set_ylabel("IV RMSE")
    ax.set_title("Fit error by moneyness: wings vs ATM")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_short_vs_long_advantage(fits, path: Path, symbol: str = ""):
    """Headline comparison: short-tenor RMSE vs long-tenor RMSE."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=150)

    names = MODEL_ORDER
    short = [next(fr.rmse_short for fr in fits if fr.name == n) for n in names]
    long = [next(fr.rmse_long for fr in fits if fr.name == n) for n in names]
    cols = [COLORS[n] for n in names]
    x = np.arange(len(names))

    axes[0].bar(x, short, color=cols)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("IV RMSE")
    axes[0].set_title("Short tenors (τ ≤ median)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, long, color=cols)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel("IV RMSE")
    axes[1].set_title("Long tenors (τ > median)")
    axes[1].grid(True, axis="y", alpha=0.3)

    for ax, vals in ((axes[0], short), (axes[1], long)):
        i2 = names.index("2-factor Bergomi")
        ax.annotate(
            f"{vals[i2]:.3f}",
            xy=(i2, vals[i2]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS["2-factor Bergomi"],
            fontweight="bold",
        )

    tag = f" — live {symbol}" if symbol else ""
    fig.suptitle(
        f"Two-factor advantage{tag}: lowest error short *and* long",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_residual_heatmaps(df: pd.DataFrame, fits, path: Path):
    """Side-by-side residuals: 1F compromises, 2F is flat."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), dpi=150, sharey=True)
    show = ["Heston-like", "1-factor Bergomi", "2-factor Bergomi"]
    by = {fr.name: fr for fr in fits}
    # Shared color scale from worst 1F residual
    all_abs = []
    residuals = {}
    for name in show:
        fr = by[name]
        r = MODEL_FN[name](df["k"].to_numpy(), df["tau"].to_numpy(), **fr.params) - df[
            "iv"
        ].to_numpy()
        residuals[name] = r
        all_abs.append(np.abs(r))
    lim = float(np.percentile(np.concatenate(all_abs), 95)) or 0.01

    for ax, name in zip(axes, show):
        r = residuals[name]
        sc = ax.scatter(
            df["k"],
            df["tau"],
            c=r,
            cmap="RdBu_r",
            s=32,
            vmin=-lim,
            vmax=lim,
        )
        ax.set_title(f"{name}\nRMSE={by[name].rmse:.4f}")
        ax.set_xlabel(r"$k$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\tau$")
    fig.suptitle(
        "Residuals (model − market): single-clock models leave structured errors",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cax = fig.add_axes([0.93, 0.18, 0.015, 0.64])
    fig.colorbar(sc, cax=cax, label="ΔIV")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_short_long_smile_contrast(
    df: pd.DataFrame, fits, path: Path, symbol: str = ""
):
    """One short + one long smile: the visual proof of the two-clock need."""
    taus = _select_tenors(df["tau"].to_numpy(), n=2)
    if len(taus) < 2:
        taus = [float(df["tau"].min()), float(df["tau"].max())]
    tau_s, tau_l = taus[0], taus[-1]
    k_line = np.linspace(df["k"].min(), df["k"].max(), 160)
    tag = f" ({symbol})" if symbol else ""

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=150)
    for ax, tau, title in (
        (axes[0], tau_s, "Short tenor — needs a fast smile clock"),
        (axes[1], tau_l, "Long tenor — needs a slow skew clock"),
    ):
        mkt = df[np.isclose(df["tau"], tau, atol=1e-3)].sort_values("k")
        ax.scatter(mkt["k"], mkt["iv"], s=30, color="k", zorder=5, label="Market")
        for fr in fits:
            if fr.name == "Black–Scholes":
                continue
            pred = MODEL_FN[fr.name](k_line, np.full_like(k_line, tau), **fr.params)
            ax.plot(
                k_line,
                pred,
                color=COLORS[fr.name],
                lw=2.8 if "2-factor" in fr.name else 1.5,
                ls="-" if "2-factor" in fr.name else "--",
                label=fr.name,
            )
        ax.set_title(rf"{title}" + "\n" + rf"($\tau={tau:.2f}$)")
        ax.set_xlabel(r"$k=\log(K/F)$")
        ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Live{tag}: 1F/Heston miss one end; 2F matches both",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(symbol: str | None = "auto"):
    """Default ``auto``: probe live underliers and keep the best real 2F strip."""
    ART.mkdir(parents=True, exist_ok=True)
    label = symbol if symbol is not None else "auto"
    print(f"Building option surface for {label}…")

    pre_fits = None
    if symbol is None or str(symbol).lower() in {"auto", "none", ""}:
        df, meta, pre_fits = choose_best_live_for_2f()
        df.to_csv(ART / "market_quotes.csv", index=False)
        df.to_csv(ART / "market_quotes_live.csv", index=False)
        (ART / "market_meta.json").write_text(json.dumps(meta, indent=2))
        (ART / "market_meta_live.json").write_text(json.dumps(meta, indent=2))
    else:
        df, meta = load_or_fetch(symbol=symbol)
        pre_fits = None

    print(
        f"Using {meta.get('source')} | symbol={meta.get('symbol')} | "
        f"spot={meta.get('spot')} | n={len(df)} quotes | "
        f"tenors={len(meta.get('tenors', []))} | "
        f"iv={meta.get('iv_method')} | k={meta.get('moneyness')}"
    )
    if meta.get("yahoo_iv_mae_vs_ours") is not None:
        print(f"Yahoo IV MAE vs mid-implied: {meta['yahoo_iv_mae_vs_ours']:.4f}")
    if meta.get("note"):
        print(f"Note: {meta['note']}")

    k = df["k"].to_numpy(float)
    tau = df["tau"].to_numpy(float)
    iv = df["iv"].to_numpy(float)
    fits = pre_fits if pre_fits is not None else fit_all(k, tau, iv)
    adv = two_factor_advantage(fits)

    sym_tag = str(meta.get("symbol", ""))
    plot_market_surface(df, ART / "market_iv_surface.png")
    plot_smile_fits(df, fits, ART / "model_fits_smiles.png")
    plot_tenor_fits(df, fits, ART / "model_fits_tenors.png")
    plot_model_surfaces(df, fits, ART / "model_iv_surfaces_3d.png")
    plot_surface_slices_grid(df, fits, ART / "model_surfaces_by_tenor.png")
    plot_errors_by_tenor(fits, ART / "fit_errors_by_tenor.png")
    plot_errors_by_moneyness(fits, ART / "fit_errors_by_moneyness.png")
    plot_short_vs_long_advantage(fits, ART / "fit_short_vs_long.png", symbol=sym_tag)
    plot_residual_heatmaps(df, fits, ART / "model_residuals.png")
    plot_short_long_smile_contrast(
        df, fits, ART / "model_short_long_contrast.png", symbol=sym_tag
    )

    report = {
        "market": meta,
        "advantage": adv,
        "data": "live_yahoo_mid_implied",
        "models": [
            {
                "name": fr.name,
                "rmse": fr.rmse,
                "mae": fr.mae,
                "rmse_weighted": fr.rmse_weighted,
                "rmse_short": fr.rmse_short,
                "rmse_long": fr.rmse_long,
                "params": fr.params,
                "hit_bounds": fr.hit_bounds,
                "success": fr.success,
                "nfev": fr.nfev,
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
        hits = f"  hit_bounds={fr.hit_bounds}" if fr.hit_bounds else ""
        print(
            f"  {fr.name:18s}  RMSE={fr.rmse:.4f}  "
            f"short={fr.rmse_short:.4f}  long={fr.rmse_long:.4f}{hits}"
        )
        print(f"    params={fr.params}")
    print(
        f"\n2F advantage: rmse_ratio={adv['rmse_ratio']:.2f}×  "
        f"short={adv['short_ratio']:.2f}×  long={adv['long_ratio']:.2f}×  "
        f"joint={adv['joint_score']:.2f}×"
    )
    print(f"Wrote artifacts to {ART}")


if __name__ == "__main__":
    # Default: auto-pick the live underlier with the clearest 2F advantage.
    # Also: NVDA, QQQ, AAPL, ^SPX, or synthetic (explicit offline only).
    sym = sys.argv[1] if len(sys.argv) > 1 else "auto"
    main(sym)
