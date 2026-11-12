#!/usr/bin/env python3
"""
Parametric IV-surface models and least-squares fits to market quotes.

Factor structure (what differs across models):

- Black–Scholes: constant σ
- Heston-like: one mean-reversion clock for ATM / skew / smile
- 1-factor Bergomi: one OU clock with free ATM/skew/smile amplitudes
- 2-factor Bergomi: fast + slow OU clocks (independent short/long structure)

All shape amplitudes are free; clocks (κ) encode the factor structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def bs_iv(k, tau, sigma=0.2):
    k = np.asarray(k, dtype=float)
    return np.full(np.shape(k), float(sigma), dtype=float)


def heston_like_iv(
    k,
    tau,
    v0=0.04,
    v_inf=0.04,
    kappa=2.0,
    skew0=-0.3,
    skew_decay=0.6,
    smile0=0.15,
    smile_decay=0.4,
):
    """Single-timescale surface: ATM → v_inf, skew/smile share one clock."""
    k = np.asarray(k, dtype=float)
    tau = np.asarray(tau, dtype=float)
    atm = np.sqrt(
        np.maximum(v_inf + (v0 - v_inf) * np.exp(-kappa * tau), 1e-8)
    )
    skew = skew0 * np.exp(-skew_decay * tau) * k
    smile = smile0 * np.exp(-smile_decay * tau) * (k**2)
    return np.clip(atm + skew + smile, 0.03, 1.5)


def bergomi1_like_iv(
    k,
    tau,
    xi0=0.04,
    kappa=2.0,
    atm0=0.0,
    skew0=-0.4,
    smile0=0.35,
    smile_decay=1.2,
):
    """One OU clock: ATM/skew ~ e^{-κτ}; smile has its own decay."""
    k = np.asarray(k, dtype=float)
    tau = np.asarray(tau, dtype=float)
    w = np.exp(-kappa * tau)
    atm = np.sqrt(max(xi0, 1e-8)) + atm0 * w
    skew = skew0 * w * k
    smile = smile0 * np.exp(-smile_decay * tau) * (k**2)
    return np.clip(atm + skew + smile, 0.03, 1.5)


def bergomi2_like_iv(
    k,
    tau,
    xi0=0.04,
    kappa1=8.0,
    kappa2=0.5,
    atm1=0.0,
    atm2=0.0,
    skew1=-0.35,
    skew2=-0.20,
    smile1=0.45,
    smile2=0.12,
):
    """Two OU clocks: fast (short smile/skew) + slow (long ATM/skew)."""
    k = np.asarray(k, dtype=float)
    tau = np.asarray(tau, dtype=float)
    w1 = np.exp(-kappa1 * tau)
    w2 = np.exp(-kappa2 * tau)
    atm = np.sqrt(max(xi0, 1e-8)) + atm1 * w1 + atm2 * w2
    skew = (skew1 * w1 + skew2 * w2) * k
    smile = (smile1 * w1 + smile2 * w2) * (k**2)
    return np.clip(atm + skew + smile, 0.03, 1.5)


@dataclass
class FitResult:
    name: str
    params: dict
    rmse: float
    mae: float
    rmse_by_tenor: dict
    rmse_by_moneyness: dict
    success: bool
    nfev: int
    hit_bounds: list


def _pack(names, x):
    return {n: float(v) for n, v in zip(names, x)}


def _rmse_by_tenor(tau: np.ndarray, err: np.ndarray) -> dict:
    by = {}
    for t in np.unique(np.round(tau, 6)):
        m = np.isclose(tau, t, atol=1e-4)
        if m.any():
            by[float(t)] = float(np.sqrt(np.mean(err[m] ** 2)))
    return by


def _rmse_by_moneyness(k: np.ndarray, err: np.ndarray, n_bins: int = 5) -> dict:
    if len(k) < n_bins:
        n_bins = max(1, len(k))
    edges = np.unique(np.quantile(k, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return {
            f"{float(k.min()):.2f}:{float(k.max()):.2f}": float(
                np.sqrt(np.mean(err**2))
            )
        }
    by = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (k >= lo) & (k <= hi if hi == edges[-1] else k < hi)
        if m.any():
            by[f"{lo:.2f}:{hi:.2f}"] = float(np.sqrt(np.mean(err[m] ** 2)))
    return by


def _hit_bounds(x, lo, hi, names, tol=1e-8):
    hits = []
    for n, xi, a, b in zip(names, x, lo, hi):
        if abs(xi - a) <= tol or abs(xi - b) <= tol:
            hits.append(n)
    return hits


def fit_model(name, fn, names, x0, bounds, k, tau, iv, n_starts: int = 6) -> FitResult:
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    x0 = np.asarray(x0, float)
    rng = np.random.default_rng(0)

    def resid(x):
        return fn(k, tau, **_pack(names, x)) - iv

    candidates = [x0]
    for _ in range(max(0, n_starts - 1)):
        jitter = x0 + 0.25 * (hi - lo) * rng.normal(size=len(x0))
        candidates.append(np.clip(jitter, lo + 1e-8, hi - 1e-8))

    best = None
    for x_init in candidates:
        sol = least_squares(
            resid,
            x0=x_init,
            bounds=(lo, hi),
            max_nfev=1200,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        cost = float(np.sum(sol.fun**2))
        if best is None or cost < best[0]:
            best = (cost, sol)

    _, sol = best
    params = _pack(names, sol.x)
    pred = fn(k, tau, **params)
    err = pred - iv
    return FitResult(
        name=name,
        params=params,
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        rmse_by_tenor=_rmse_by_tenor(tau, err),
        rmse_by_moneyness=_rmse_by_moneyness(k, err),
        success=bool(sol.success),
        nfev=int(sol.nfev),
        hit_bounds=_hit_bounds(sol.x, lo, hi, names),
    )


def fit_all(k, tau, iv) -> list[FitResult]:
    atm0 = (
        float(np.median(iv[np.abs(k) < 0.03]))
        if np.any(np.abs(k) < 0.03)
        else float(np.median(iv))
    )
    v0 = max(atm0**2, 0.01)
    return [
        fit_model(
            "Black–Scholes",
            bs_iv,
            ["sigma"],
            [atm0],
            ([0.05], [0.8]),
            k,
            tau,
            iv,
            n_starts=1,
        ),
        fit_model(
            "Heston-like",
            heston_like_iv,
            ["v0", "v_inf", "kappa", "skew0", "skew_decay", "smile0", "smile_decay"],
            [v0, v0, 2.0, -0.5, 0.7, 1.0, 1.0],
            (
                [0.002, 0.002, 0.05, -3.0, 0.02, 0.01, 0.02],
                [0.40, 0.40, 20.0, 0.5, 8.0, 8.0, 10.0],
            ),
            k,
            tau,
            iv,
        ),
        fit_model(
            "1-factor Bergomi",
            bergomi1_like_iv,
            ["xi0", "kappa", "atm0", "skew0", "smile0", "smile_decay"],
            [v0, 2.0, 0.02, -0.5, 1.0, 1.5],
            (
                [0.002, 0.05, -0.15, -3.0, 0.01, 0.05],
                [0.40, 20.0, 0.25, 0.5, 8.0, 10.0],
            ),
            k,
            tau,
            iv,
        ),
        fit_model(
            "2-factor Bergomi",
            bergomi2_like_iv,
            [
                "xi0",
                "kappa1",
                "kappa2",
                "atm1",
                "atm2",
                "skew1",
                "skew2",
                "smile1",
                "smile2",
            ],
            [v0, 8.0, 0.5, 0.03, 0.01, -0.6, -0.25, 1.5, 0.4],
            (
                [0.002, 0.8, 0.01, -0.20, -0.15, -4.0, -3.0, 0.01, 0.0],
                [0.40, 40.0, 4.0, 0.30, 0.25, 0.5, 0.5, 10.0, 5.0],
            ),
            k,
            tau,
            iv,
        ),
    ]


MODEL_FN = {
    "Black–Scholes": bs_iv,
    "Heston-like": heston_like_iv,
    "1-factor Bergomi": bergomi1_like_iv,
    "2-factor Bergomi": bergomi2_like_iv,
}
