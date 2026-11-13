#!/usr/bin/env python3
"""Greeks plots for vanilla Europeans and down-and-out barriers.

Units (all models, comparable):
  - Delta / Gamma: ∂V/∂S, ∂²V/∂S²
  - Theta: calendar decay per day  (∂V/∂t with t in days)
  - Vega: per +1 percentage point of spot vol  (dσ = 0.01)
"""

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
from stochastic_volatility_demo.output_layout import demo_dir, model_dir

COLORS = {
    "Black–Scholes": "0.35",
    "Heston-like": "C1",
    "1-factor Bergomi": "C2",
    "2-factor Bergomi": "C3",
}

# Reporting scales
_THETA_PER_DAY = 1.0 / 365.0
_VEGA_PER_PCT = 0.01  # absolute vol bump for "per 1 vol point"


def _bs_price_greeks(S, K, T, r, q, sigma):
    """Closed-form European call; Vega = ∂V/∂σ, Theta = ∂V/∂t (calendar year)."""
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


def _scale_report(theta_annual, vega_d_sigma):
    """Convert to theta/day and vega per +1 vol point."""
    return np.asarray(theta_annual) * _THETA_PER_DAY, np.asarray(vega_d_sigma) * _VEGA_PER_PCT


def _spot_vol(fit_name: str, params: dict) -> float:
    if fit_name == "Black–Scholes":
        return float(params["sigma"])
    if fit_name == "Heston-like":
        return float(np.sqrt(max(params["v0"], 1e-12)))
    return float(np.sqrt(max(params["xi0"], 1e-12)))


def _bump_spot_vol_abs(fit_name: str, params: dict, d_sigma: float) -> dict:
    """Bump the instantaneous / spot vol by an absolute amount d_sigma."""
    p = dict(params)
    if fit_name == "Black–Scholes":
        p["sigma"] = float(params["sigma"]) + d_sigma
    elif fit_name == "Heston-like":
        s0 = float(np.sqrt(max(params["v0"], 1e-12)))
        s_inf = float(np.sqrt(max(params["v_inf"], 1e-12)))
        p["v0"] = (s0 + d_sigma) ** 2
        p["v_inf"] = (s_inf + d_sigma) ** 2
    else:
        s0 = float(np.sqrt(max(params["xi0"], 1e-12)))
        p["xi0"] = (s0 + d_sigma) ** 2
    return p


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
    n_paths: int = 40_000,
    n_steps: int = 64,
    seed: int = 3,
):
    # Wide enough window to show the BS gamma hump at equity vols
    if spot_grid is None:
        spot_grid = np.linspace(S0 * 0.70, S0 * 1.30, 25)

    if fit_name == "Black–Scholes":
        sigma = float(params["sigma"])
        price, delta, gamma, theta_a, vega_s = _bs_price_greeks(
            spot_grid, K, T, r, q, sigma
        )
        theta, vega = _scale_report(theta_a, vega_s)
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
            "units": {"theta": "per day", "vega": "per +1 vol point"},
        }

    # Common-seed MC prices across the spot grid, then spline Δ/Γ.
    # Same seed for vol/time bumps → CRN; spline removes residual MC jitter.
    from scipy.interpolate import UnivariateSpline

    dT = 1.0 / 365.0
    d_sigma = 0.01  # +1 vol point
    prices, thetas, vegas = [], [], []

    for S in spot_grid:
        mid = _euro_mc_price(
            fit_name, params, S, K, T, r, q, n_paths, n_steps, seed
        )
        prices.append(mid)
        tm = _euro_mc_price(
            fit_name, params, S, K, max(T - dT, 1e-4), r, q, n_paths, n_steps, seed
        )
        thetas.append(tm - mid)  # per day
        vv = _euro_mc_price(
            fit_name,
            _bump_spot_vol_abs(fit_name, params, d_sigma),
            S,
            K,
            T,
            r,
            q,
            n_paths,
            n_steps,
            seed,
        )
        vegas.append(vv - mid)

    prices = np.asarray(prices, float)
    thetas = np.asarray(thetas, float)
    vegas = np.asarray(vegas, float)

    # Smooth price → analytic Δ/Γ; light smooth on Θ/Vega curves
    s_price = max(1e-8, 0.02 * len(spot_grid) * float(np.var(prices) + 1e-12))
    price_spl = UnivariateSpline(spot_grid, prices, k=3, s=s_price)
    delta = price_spl.derivative(1)(spot_grid)
    gamma = price_spl.derivative(2)(spot_grid)
    s_tv = max(1e-8, 0.05 * len(spot_grid) * float(np.var(thetas) + 1e-12))
    theta = UnivariateSpline(spot_grid, thetas, k=3, s=s_tv)(spot_grid)
    s_vv = max(1e-8, 0.05 * len(spot_grid) * float(np.var(vegas) + 1e-12))
    vega = UnivariateSpline(spot_grid, vegas, k=3, s=s_vv)(spot_grid)

    i0 = int(np.argmin(np.abs(spot_grid - S0)))
    return {
        "S": spot_grid,
        "price": price_spl(spot_grid),
        "delta": np.asarray(delta),
        "gamma": np.asarray(gamma),
        "theta": np.asarray(theta),
        "vega": np.asarray(vega),
        "spot0": S0,
        "delta0": float(delta[i0]),
        "gamma0": float(gamma[i0]),
        "theta0": float(theta[i0]),
        "vega0": float(vega[i0]),
        "price0": float(price_spl(S0)),
        "units": {"theta": "per day", "vega": "per +1 vol point"},
        "spot_vol": _spot_vol(fit_name, params),
    }


def _save_model_greek_panel(
    name: str,
    S: np.ndarray,
    series: dict,
    path: Path,
    title: str,
    S0: float | None = None,
    K: float | None = None,
    B: float | None = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), dpi=140)
    specs = [
        ("delta", "Delta", axes[0, 0]),
        ("gamma", "Gamma", axes[0, 1]),
        ("theta", "Theta (per day)", axes[1, 0]),
        ("vega", "Vega (per +1 vol pt)", axes[1, 1]),
    ]
    for key, ylab, ax in specs:
        ax.plot(S, series[key], color=COLORS.get(name, "C0"), lw=2.0)
        if S0 is not None:
            ax.axvline(S0, color="0.55", ls=":", lw=1.0, label="spot")
        if K is not None:
            ax.axvline(K, color="C0", ls="--", lw=0.9, alpha=0.7, label="K")
        if B is not None:
            ax.axvline(B, color="C3", ls="--", lw=0.9, alpha=0.7, label="B")
        ax.set_xlabel("Spot S")
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        ax.grid(True, alpha=0.3)
    # one legend on first panel if markers present
    if S0 is not None or K is not None or B is not None:
        axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_comparison(
    greeks_by_model: dict,
    path: Path,
    greek_key: str,
    ylabel: str,
    title: str,
    S0: float | None = None,
    K: float | None = None,
    B: float | None = None,
):
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=140)
    for name, g in greeks_by_model.items():
        S = g["S"] if isinstance(g, dict) else g.S
        y = g[greek_key] if isinstance(g, dict) else getattr(g, greek_key)
        ax.plot(
            S,
            y,
            color=COLORS.get(name, "C0"),
            lw=2.3 if "2-factor" in name else 1.5,
            label=name,
        )
    if S0 is not None:
        ax.axvline(S0, color="0.5", ls=":", lw=1.0)
    if K is not None:
        ax.axvline(K, color="C0", ls="--", lw=0.9, alpha=0.6)
    if B is not None:
        ax.axvline(B, color="C3", ls="--", lw=0.9, alpha=0.6)
    ax.set_xlabel("Spot S")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_vanilla_greeks_all(
    greeks_by_model: dict,
    symbol: str,
    S0: float,
    K: float | None = None,
    out_root: Path | None = None,
):
    """One panel per model + four comparison curves under ``results/sv_demo/vanilla``."""
    for name, g in greeks_by_model.items():
        d = model_dir("vanilla", name, out_root)
        _save_model_greek_panel(
            name,
            g["S"],
            g,
            d / "greeks.png",
            title=f"{symbol} | {name} | European call Greeks",
            S0=S0,
            K=K,
        )

    cmp = demo_dir("vanilla", out_root)
    for key, label in (
        ("delta", "Delta"),
        ("gamma", "Gamma"),
        ("theta", "Theta (per day)"),
        ("vega", "Vega (per +1 vol pt)"),
    ):
        _save_comparison(
            greeks_by_model,
            cmp / f"greeks_{key}.png",
            key,
            label,
            f"{symbol}: European call {label} by model",
            S0=S0,
            K=K,
        )


def plot_barrier_greeks_all(
    greeks_by_model: dict[str, BarrierGreeks],
    symbol: str,
    S0: float,
    K: float | None = None,
    B: float | None = None,
    out_root: Path | None = None,
):
    """Per-model panels + comparison curves under ``results/sv_demo/barrier``."""
    for name, g in greeks_by_model.items():
        d = model_dir("barrier", name, out_root)
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
            title=f"{symbol} | {name} | Down-and-out call Greeks",
            S0=S0,
            K=K,
            B=B,
        )

    cmp = demo_dir("barrier", out_root)
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
        ("theta", "Theta (per day)"),
        ("vega", "Vega (per +1 vol pt)"),
    ):
        _save_comparison(
            as_dict,
            cmp / f"greeks_{key}.png",
            key,
            label,
            f"{symbol}: Down-and-out call {label} by model",
            S0=S0,
            K=K,
            B=B,
        )
