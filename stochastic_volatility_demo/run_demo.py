#!/usr/bin/env python3
"""
Exotic pricing demo — American down-and-out call (no closed form).

Narrative
---------
1. Calibrate BS / Heston / 1F Bergomi / 2F Bergomi to **live** vanilla IVs
2. Map each fit to risk-neutral dynamics
3. Price the **same** American down-and-out call under each model via
   Longstaff–Schwartz Monte Carlo (path-dependent knock-out + early exercise)
4. Show that model choice changes the exotic price — there is no analytical
   reference; the 2-factor Bergomi surface (best vanilla fit) is the anchor

This is the point of PINN / numerical SV pricing: exotics without analytics.
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

from stochastic_volatility_demo.american_barrier import price_american_down_out_call
from stochastic_volatility_demo.fetch_market import (
    choose_best_live_for_2f,
    filter_iv_outliers,
    load_or_fetch,
)
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


def _clean_market(df, meta, art: Path):
    of = meta.get("outlier_filter") or {}
    if of.get("n_removed"):
        print(
            f"Outlier filter (fetch): removed {of['n_removed']} / {of.get('n_before')} "
            f"(kept {of.get('n_after')})"
        )
    removed = list(meta.get("outliers_removed") or [])
    gate_removed = 0
    for _ in range(4):
        df_clean, final_info = filter_iv_outliers(
            df, z_max=2.5, max_abs_resid=0.035, max_neighbor_jump=0.05
        )
        if final_info["n_removed"] == 0:
            df = df_clean
            break
        gate_removed += final_info["n_removed"]
        removed.extend(final_info["removed"])
        df = df_clean
    if gate_removed:
        print(f"Outlier filter (final): removed {gate_removed} more → n={len(df)}")
        of = {
            **of,
            "n_after": int(len(df)),
            "n_removed": int(of.get("n_removed", 0) + gate_removed),
        }
        meta["outlier_filter"] = of
        meta["outliers_removed"] = removed
    df = df.reset_index(drop=True)
    df.to_csv(art / "market_quotes.csv", index=False)
    if removed:
        pd.DataFrame(removed).to_csv(art / "outliers_removed.csv", index=False)
    return df, meta


# ---------------------------------------------------------------------------
# Plots — surface support
# ---------------------------------------------------------------------------


def plot_smile_support(df, fits, path: Path, symbol: str):
    tenors = _select_tenors(df["tau"].to_numpy(), n=2)
    if len(tenors) < 2:
        tenors = [float(df["tau"].min()), float(df["tau"].max())]
    k_line = np.linspace(df["k"].min(), df["k"].max(), 160)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=150)
    for ax, tau in zip(axes, (tenors[0], tenors[-1])):
        mkt = df[np.isclose(df["tau"], tau, atol=1e-3)].sort_values("k")
        ax.scatter(mkt["k"], mkt["iv"], s=22, color="k", zorder=5, label="Market")
        for fr in fits:
            pred = MODEL_FN[fr.name](k_line, np.full_like(k_line, tau), **fr.params)
            ax.plot(
                k_line,
                pred,
                color=COLORS[fr.name],
                lw=2.4 if "2-factor" in fr.name else 1.3,
                ls="-" if "2-factor" in fr.name else "--",
                label=fr.name,
            )
        ax.set_title(rf"Vanilla smile $\tau={tau:.2f}$")
        ax.set_xlabel(r"$k=\log(K/F)$")
        ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"{symbol}: calibrate vanillas first (2F usually wins short+long)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fit_rmse_bars(fits, path: Path, symbol: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    names = [fr.name for fr in fits]
    xs = np.arange(len(names))
    ax.bar(xs, [fr.rmse for fr in fits], color=[COLORS[n] for n in names])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Vanilla IV RMSE")
    ax.set_title(f"{symbol}: surface fit quality (input to exotic dynamics)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — American barrier headline
# ---------------------------------------------------------------------------


def plot_barrier_prices(quotes, path: Path, symbol: str, contract: dict):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), dpi=150)
    names = [q.name for q in quotes]
    x = np.arange(len(names))
    cols = [COLORS[n] for n in names]

    axes[0].bar(x, [q.price for q in quotes], color=cols, yerr=[q.stderr for q in quotes],
                capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("Price")
    axes[0].set_title("American down-and-out call")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, [q.european_barrier for q in quotes], color=cols, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel("Price")
    axes[1].set_title("European down-and-out (no exercise)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, [q.knock_prob for q in quotes], color=cols)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=15)
    axes[2].set_ylabel("Probability")
    axes[2].set_title("Knock-out probability")
    axes[2].grid(True, axis="y", alpha=0.3)

    B, K, T = contract["B"], contract["K"], contract["T"]
    fig.suptitle(
        f"{symbol}: American DAO call  K={K:.1f}, B={B:.1f}, T={T:.2f}y  — no closed form",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_american_vs_european(quotes, path: Path, symbol: str):
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    names = [q.name for q in quotes]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(
        x - w / 2,
        [q.european_barrier for q in quotes],
        width=w,
        color=[COLORS[n] for n in names],
        alpha=0.45,
        label="European barrier",
    )
    ax.bar(
        x + w / 2,
        [q.price for q in quotes],
        width=w,
        color=[COLORS[n] for n in names],
        label="American barrier",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Price")
    ax.set_title(f"{symbol}: early-exercise premium on the barrier call")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_price_vs_barrier(series_by_model: dict, path: Path, symbol: str, S0: float):
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    for name, rows in series_by_model.items():
        Bs = [r["B"] for r in rows]
        ps = [r["price"] for r in rows]
        ax.plot(
            Bs,
            ps,
            "o-",
            color=COLORS[name],
            lw=2.4 if "2-factor" in name else 1.4,
            label=name,
        )
    ax.axvline(S0, color="0.5", ls=":", lw=1.0, label=f"spot={S0:.1f}")
    ax.set_xlabel("Barrier B")
    ax.set_ylabel("American DAO call price")
    ax.set_title(f"{symbol}: price vs barrier — model disagreement grows near spot")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_price_vs_maturity(series_by_model: dict, path: Path, symbol: str):
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    for name, rows in series_by_model.items():
        Ts = [r["T"] for r in rows]
        ps = [r["price"] for r in rows]
        ax.plot(
            Ts,
            ps,
            "o-",
            color=COLORS[name],
            lw=2.4 if "2-factor" in name else 1.4,
            label=name,
        )
    ax.set_xlabel(r"Maturity $\tau$")
    ax.set_ylabel("American DAO call price")
    ax.set_title(f"{symbol}: price vs maturity under each calibrated dynamics")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_model_spread(quotes, path: Path, symbol: str):
    """Highlight 2F vs single-clock exotic price gap."""
    by = {q.name: q for q in quotes}
    f2 = by["2-factor Bergomi"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=150)
    names = [n for n in MODEL_ORDER if n != "2-factor Bergomi"]
    diffs = [by[n].price - f2.price for n in names]
    ax.bar(np.arange(len(names)), diffs, color=[COLORS[n] for n in names])
    ax.axhline(0.0, color="0.3", lw=1.0)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Price − 2F Bergomi")
    ax.set_title(
        f"{symbol}: exotic mispricing vs 2F anchor (best vanilla surface)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(symbol: str | None = "auto"):
    ART.mkdir(parents=True, exist_ok=True)
    label = symbol if symbol is not None else "auto"
    print(f"=== American barrier demo on {label} ===")

    if symbol is None or str(symbol).lower() in {"auto", "none", ""}:
        df, meta, _ = choose_best_live_for_2f()
    else:
        df, meta = load_or_fetch(symbol=symbol)

    df, meta = _clean_market(df, meta, ART)
    sym = str(meta.get("symbol", label))
    spot = float(meta.get("spot") or df["spot"].iloc[0])
    r = float(meta.get("r", 0.04))
    q = float(meta.get("q_fallback") or meta.get("q") or 0.005)

    print(
        f"Market: {meta.get('source')} | {sym} | spot={spot:.4f} | "
        f"n={len(df)} cleaned quotes | tenors={len(meta.get('tenors', []))}"
    )

    # --- 1. Calibrate vanillas ------------------------------------------------
    fits = fit_all(
        df["k"].to_numpy(float),
        df["tau"].to_numpy(float),
        df["iv"].to_numpy(float),
    )
    adv = two_factor_advantage(fits)
    print("\nVanilla IV RMSE:")
    for fr in sorted(fits, key=lambda z: z.rmse):
        print(
            f"  {fr.name:18s}  RMSE={fr.rmse:.4f}  "
            f"short={fr.rmse_short:.4f}  long={fr.rmse_long:.4f}"
        )
    print(
        f"2F vanilla advantage: joint={adv['joint_score']:.2f}×  "
        f"rmse_ratio={adv['rmse_ratio']:.2f}×"
    )

    plot_smile_support(df, fits, ART / "vanilla_smiles.png", sym)
    plot_fit_rmse_bars(fits, ART / "vanilla_fit_rmse.png", sym)

    # --- 2. American down-and-out contract ------------------------------------
    # Slightly OTM call, barrier ~10% below spot, maturity ~ mid surface tenor
    tenors = sorted(float(t) for t in df["tau"].unique())
    T = float(tenors[min(len(tenors) - 1, max(1, len(tenors) // 2))])
    K = round(spot * 1.05, 2)
    B = round(spot * 0.90, 2)
    contract = {"S0": spot, "K": K, "B": B, "T": T, "r": r, "q": q}
    print(
        f"\nExotic: American down-and-out CALL  "
        f"S0={spot:.2f} K={K:.2f} B={B:.2f} T={T:.3f}y  (no analytical price)"
    )

    n_paths, n_steps = 20_000, 48
    quotes = []
    for fr in fits:
        print(f"  Pricing under {fr.name}…")
        qte = price_american_down_out_call(
            fr.name,
            fr.params,
            S0=spot,
            K=K,
            B=B,
            T=T,
            r=r,
            q=q,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=42,
        )
        quotes.append(qte)
        print(
            f"    American={qte.price:.4f}±{qte.stderr:.4f}  "
            f"EuroBarrier={qte.european_barrier:.4f}  "
            f"EuroVanilla={qte.european_vanilla:.4f}  "
            f"KO={qte.knock_prob:.1%}  EEP={qte.early_exercise_premium:.4f}"
        )

    plot_barrier_prices(quotes, ART / "american_barrier_prices.png", sym, contract)
    plot_american_vs_european(quotes, ART / "american_vs_european_barrier.png", sym)
    plot_model_spread(quotes, ART / "american_barrier_vs_2f.png", sym)

    # --- 3. Sensitivities -----------------------------------------------------
    barriers = np.linspace(spot * 0.80, spot * 0.95, 5)
    by_fit = {fr.name: fr for fr in fits}
    series_B = {}
    print("\nPrice vs barrier…")
    for name in MODEL_ORDER:
        rows = []
        for i, Bb in enumerate(barriers):
            qte = price_american_down_out_call(
                name,
                by_fit[name].params,
                S0=spot,
                K=K,
                B=float(Bb),
                T=T,
                r=r,
                q=q,
                n_paths=12_000,
                n_steps=40,
                seed=100 + i,
            )
            rows.append({"B": float(Bb), "price": qte.price, "knock_prob": qte.knock_prob})
        series_B[name] = rows
    plot_price_vs_barrier(series_B, ART / "american_barrier_vs_B.png", sym, spot)

    mats = np.array(sorted(set([tenors[0], T, tenors[-1]])))
    series_T = {}
    print("Price vs maturity…")
    for name in MODEL_ORDER:
        rows = []
        for i, TT in enumerate(mats):
            qte = price_american_down_out_call(
                name,
                by_fit[name].params,
                S0=spot,
                K=K,
                B=B,
                T=float(TT),
                r=r,
                q=q,
                n_paths=12_000,
                n_steps=max(24, int(40 * TT / max(T, 1e-6))),
                seed=200 + i,
            )
            rows.append({"T": float(TT), "price": qte.price, "knock_prob": qte.knock_prob})
        series_T[name] = rows
    plot_price_vs_maturity(series_T, ART / "american_barrier_vs_T.png", sym)

    # --- Report ---------------------------------------------------------------
    report = {
        "exercise": "american_down_and_out_call",
        "note": (
            "No analytical price. Each model is calibrated to live vanillas, "
            "then used as risk-neutral dynamics for LSMC barrier pricing."
        ),
        "market": meta,
        "vanilla_advantage": adv,
        "contract": contract,
        "mc": {"n_paths": n_paths, "n_steps": n_steps},
        "vanilla_fits": [
            {
                "name": fr.name,
                "rmse": fr.rmse,
                "rmse_short": fr.rmse_short,
                "rmse_long": fr.rmse_long,
                "params": fr.params,
                "hit_bounds": fr.hit_bounds,
            }
            for fr in fits
        ],
        "american_barrier": [
            {
                "name": q.name,
                "price": q.price,
                "stderr": q.stderr,
                "european_barrier": q.european_barrier,
                "european_vanilla": q.european_vanilla,
                "knock_prob": q.knock_prob,
                "early_exercise_premium": q.early_exercise_premium,
                "dynamics": q.params_used,
            }
            for q in quotes
        ],
        "price_vs_barrier": series_B,
        "price_vs_maturity": series_T,
        "ranking_vanilla": [fr.name for fr in sorted(fits, key=lambda z: z.rmse)],
        "ranking_exotic": [q.name for q in sorted(quotes, key=lambda z: z.price)],
    }
    (ART / "fit_report.json").write_text(json.dumps(report, indent=2))
    (ART / "market_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nAmerican DAO prices (no closed form):")
    for q in sorted(quotes, key=lambda z: -z.price):
        print(f"  {q.name:18s}  {q.price:.4f}  (KO={q.knock_prob:.1%})")
    print(f"\nWrote artifacts to {ART}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "auto"
    main(sym)
