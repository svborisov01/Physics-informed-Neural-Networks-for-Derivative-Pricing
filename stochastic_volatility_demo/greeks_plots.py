#!/usr/bin/env python3
"""Greeks plots for vanilla Europeans and American barrier under each model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from stochastic_volatility_demo.american_barrier import (
    BarrierGreeks,
    dynamics_from_fit,
    simulate_bergomi1,
    simulate_bergomi2,
    simulate_bs,
    simulate_heston,
)
from stochastic_volatility_demo.artifacts_layout import comparison_dir, model_graph_dir

COLORS = {
    "Black–Scholes": "0.35",
    "Heston-like": "C1",
    "1-factor Bergomi": "C2",
    "2-factor Bergomi": "C3",
}


# ---------------------------------------------------------------------------
# Vanilla European Greeks (BS closed form; others via FD on European MC)
# ---------------------------------------------------------------------------


def _bs_price_greeks(S, K, T, r, q, sigma):
    S = np.asarray(S, dtype=float)
    vol = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol
    d2 = d1 - vol
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    price = disc_q * S * norm.cdf(d1) - disc_r * K * norm.cdf(d2)
    delta = disc_q * norm.cdf(d1)
    gamma = disc_q * norm.pdf(d1) / (S * vol)
    theta = (
        -disc_q * S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
        + q * disc_q * S * norm.cdf(d1)
        - r * disc_r * K * norm.cdf(d2)
    )
    vega = disc_q * S * norm.pdf(d1) * np.sqrt(T)
    return price, delta, gamma, theta, vega


def _euro_mc_price(fit_name, params, S0, K, T, r, q, n_paths, n_steps, seed):
    dyn = dynamics_from_fit(fit_name, params, S0)
    m = dyn["model"]
    if m == "bs":
        S = simulate_bs(S0, r, q, dyn["sigma"], T, n_paths, n_steps, seed)
    elif m == "heston":
        S = simulate_heston(
            S0, r, q, dyn["v0"], dyn["kappa"], dyn["theta"], dyn["xi"], dyn["rho"],
            T, n_paths, n_steps, seed,
        )
    elif m == "bergomi1":
        S = simulate_bergomi1(
            S0, r, q, dyn["xi0"], dyn["omega"], dyn["kappa"], dyn["rho"],
            T, n_paths, n_steps, seed,
        )
    else:
        S = simulate_bergomi2(
            S0, r, q, dyn["xi0"], dyn["omega1"], dyn["omega2"], dyn["kappa1"],
            dyn["kappa2"], dyn["rho1"], dyn["rho2"], dyn["rho12"],
            T, n_paths, n_steps, seed,
        )
    payoff = np.maximum(S[-1] - K, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


def compute_vanilla_greeks_grid(
    fit_name: str,
    params: dict,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    spot_grid: np.ndarray | None = None,
    n_paths: int = 10_000,
    n_steps: int = 40,
    seed: int = 3,
):
    if spot_grid is None:
        spot_grid = np.linspace(S0 * 0.85, S0 * 1.15, 13)

    if fit_name == "Black–Scholes":
        sigma = float(params["sigma"])
        price, delta, gamma, theta, vega = _bs_price_greeks(
            spot_grid, K, T, r, q, sigma
        )
        i0 = int(np.argmin(np.abs(spot_grid - S0)))
        return {
            "S": spot_grid,
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "spot0": S0,
            "delta0": float(delta[i0]),
            "gamma0": float(gamma[i0]),
            "theta0": float(theta[i0]),
            "vega0": float(vega[i0]),
            "price0": float(price[i0]),
        }

    # FD on European MC for SV models
    dS_rel, dT, dVol = 0.01, 1.0 / 365.0, 0.05
    prices, deltas, gammas, thetas, vegas = [], [], [], [], []

    def bump_params(bump):
        p = dict(params)
        if fit_name == "Heston-like":
            p["v0"] *= 1.0 + bump
            p["v_inf"] *= 1.0 + bump
        else:
            p["xi0"] = float(params["xi0"]) * (1.0 + bump)
        return p

    for i, S in enumerate(spot_grid):
        h = max(dS_rel * S, 1e-4 * S0)
        s_i = seed + 17 * i
        up = _euro_mc_price(
            fit_name, params, S + h, K, T, r, q, n_paths, n_steps, s_i
        )
        mid = _euro_mc_price(
            fit_name, params, S, K, T, r, q, n_paths, n_steps, s_i
        )
        dn = _euro_mc_price(
            fit_name, params, S - h, K, T, r, q, n_paths, n_steps, s_i
        )
        prices.append(mid)
        deltas.append((up - dn) / (2 * h))
        gammas.append((up - 2 * mid + dn) / (h * h))
        tm = _euro_mc_price(
            fit_name,
            params,
            S,
            K,
            max(T - dT, 1e-4),
            r,
            q,
            n_paths,
            n_steps,
            s_i,
        )
        thetas.append((tm - mid) / dT)
        vv = _euro_mc_price(
            fit_name,
            bump_params(dVol),
            S,
            K,
            T,
            r,
            q,
            n_paths,
            n_steps,
            s_i,
        )
        vegas.append((vv - mid) / dVol)

    prices = np.asarray(prices)
    i0 = int(np.argmin(np.abs(spot_grid - S0)))
    return {
        "S": spot_grid,
        "price": prices,
        "delta": np.asarray(deltas),
        "gamma": np.asarray(gammas),
        "theta": np.asarray(thetas),
        "vega": np.asarray(vegas),
        "spot0": S0,
        "delta0": float(deltas[i0]),
        "gamma0": float(gammas[i0]),
        "theta0": float(thetas[i0]),
        "vega0": float(vegas[i0]),
        "price0": float(prices[i0]),
    }


def _save_model_greek_panel(
    name: str,
    S: np.ndarray,
    series: dict,
    path: Path,
    title: str,
    S0: float | None = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), dpi=140)
    specs = [
        ("delta", "Delta", axes[0, 0]),
        ("gamma", "Gamma", axes[0, 1]),
        ("theta", "Theta", axes[1, 0]),
        ("vega", "Vega (rel. vol bump)", axes[1, 1]),
    ]
    for key, ylab, ax in specs:
        ax.plot(S, series[key], color=COLORS.get(name, "C0"), lw=2.0)
        if S0 is not None:
            ax.axvline(S0, color="0.6", ls=":", lw=1.0)
        ax.set_xlabel("Spot S")
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_comparison(
    greeks_by_model: dict,
    path: Path,
    greek_key: str,
    ylabel: str,
    title: str,
    S0: float | None = None,
):
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=140)
    for name, g in greeks_by_model.items():
        S = g["S"] if isinstance(g, dict) else g.S
        y = g[greek_key] if isinstance(g, dict) else getattr(g, greek_key)
        ax.plot(
            S,
            y,
            color=COLORS.get(name, "C0"),
            lw=2.3 if "2-factor" in name else 1.4,
            label=name,
        )
    if S0 is not None:
        ax.axvline(S0, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel("Spot S")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_vanilla_greeks_all(
    greeks_by_model: dict,
    symbol: str,
    S0: float,
    art_root: Path | None = None,
):
    """Write per-model panels + comparison curves for vanilla Europeans."""
    for name, g in greeks_by_model.items():
        d = model_graph_dir("vanilla", name, art_root)
        _save_model_greek_panel(
            name,
            g["S"],
            g,
            d / "greeks.png",
            title=f"{symbol} | {name} | European call Greeks",
            S0=S0,
        )
        # Individual greek files
        for key, label in (
            ("delta", "Delta"),
            ("gamma", "Gamma"),
            ("theta", "Theta"),
            ("vega", "Vega"),
        ):
            fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=140)
            ax.plot(g["S"], g[key], color=COLORS.get(name, "C0"), lw=2.0)
            ax.axvline(S0, color="0.6", ls=":", lw=1.0)
            ax.set_xlabel("Spot S")
            ax.set_ylabel(label)
            ax.set_title(f"{name} — {label}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(d / f"{key}.png", bbox_inches="tight", facecolor="white")
            plt.close(fig)

    cmp = comparison_dir("vanilla", art_root)
    for key, label in (
        ("delta", "Delta"),
        ("gamma", "Gamma"),
        ("theta", "Theta"),
        ("vega", "Vega (rel. vol)"),
    ):
        _save_comparison(
            greeks_by_model,
            cmp / f"greeks_{key}.png",
            key,
            label,
            f"{symbol}: European call {label} by model",
            S0=S0,
        )


def plot_barrier_greeks_all(
    greeks_by_model: dict[str, BarrierGreeks],
    symbol: str,
    S0: float,
    art_root: Path | None = None,
):
    """Write per-model panels + comparison curves for American DAO Greeks."""
    for name, g in greeks_by_model.items():
        d = model_graph_dir("barrier", name, art_root)
        series = {
            "delta": g.delta,
            "gamma": g.gamma,
            "theta": g.theta,
            "vega": g.vega,
        }
        _save_model_greek_panel(
            name,
            g.S,
            series,
            d / "greeks.png",
            title=f"{symbol} | {name} | American DAO call Greeks",
            S0=S0,
        )
        for key, label in (
            ("delta", "Delta"),
            ("gamma", "Gamma"),
            ("theta", "Theta"),
            ("vega", "Vega"),
        ):
            fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=140)
            ax.plot(g.S, getattr(g, key), color=COLORS.get(name, "C0"), lw=2.0)
            ax.axvline(S0, color="0.6", ls=":", lw=1.0)
            ax.set_xlabel("Spot S")
            ax.set_ylabel(label)
            ax.set_title(f"{name} — American DAO {label}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(d / f"{key}.png", bbox_inches="tight", facecolor="white")
            plt.close(fig)

    cmp = comparison_dir("barrier", art_root)
    as_dict = {
        n: {
            "S": g.S,
            "delta": g.delta,
            "gamma": g.gamma,
            "theta": g.theta,
            "vega": g.vega,
        }
        for n, g in greeks_by_model.items()
    }
    for key, label in (
        ("delta", "Delta"),
        ("gamma", "Gamma"),
        ("theta", "Theta"),
        ("vega", "Vega (rel. vol)"),
    ):
        _save_comparison(
            as_dict,
            cmp / f"greeks_{key}.png",
            key,
            label,
            f"{symbol}: American DAO {label} by model",
            S0=S0,
        )
