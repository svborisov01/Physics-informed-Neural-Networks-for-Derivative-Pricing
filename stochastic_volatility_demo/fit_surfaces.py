#!/usr/bin/env python3
"""
Parametric IV-surface models and least-squares fits to market quotes.

Factor structure is the *only* degree-of-freedom difference:

- Black–Scholes: flat σ
- Heston-like: **one** mean-reversion clock shared by ATM / skew / smile
- 1-factor Bergomi: **one** OU clock shared by ATM / skew / smile
- 2-factor Bergomi: **two** OU clocks (fast short smile + slow long skew)

Earlier free smile/skew decays on 1F/Heston secretly added extra clocks and
hid the 2F advantage. Residuals are **equal-weighted by tenor** so dense
short strips cannot dominate the objective.
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
    smile0=0.15,
):
    """Single clock κ drives ATM, skew, and smile jointly."""
    k = np.asarray(k, dtype=float)
    tau = np.asarray(tau, dtype=float)
    w = np.exp(-kappa * tau)
    atm = np.sqrt(np.maximum(v_inf + (v0 - v_inf) * w, 1e-8))
    skew = skew0 * w * k
    smile = smile0 * w * (k**2)
    return np.clip(atm + skew + smile, 0.03, 1.5)


def bergomi1_like_iv(
    k,
    tau,
    xi0=0.04,
    kappa=2.0,
    atm0=0.0,
    skew0=-0.4,
    smile0=0.35,
):
    """One OU clock: ATM / skew / smile all ∝ e^{-κτ}."""
    k = np.asarray(k, dtype=float)
    tau = np.asarray(tau, dtype=float)
    w = np.exp(-kappa * tau)
    atm = np.sqrt(max(xi0, 1e-8)) + atm0 * w
    skew = skew0 * w * k
    smile = smile0 * w * (k**2)
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
    """Two OU clocks: fast (short smile) + slow (long skew/ATM)."""
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
    rmse_weighted: float
    rmse_short: float
    rmse_long: float
    rmse_by_tenor: dict
    rmse_by_moneyness: dict
    success: bool
    nfev: int
    hit_bounds: list


def _pack(names, x):
    return {n: float(v) for n, v in zip(names, x)}


def _tenor_weights(tau: np.ndarray) -> np.ndarray:
    """Equal weight per unique tenor (√n_t in residual → equal tenor SSE)."""
    tau_r = np.round(tau, 6)
    w = np.ones_like(tau, dtype=float)
    for t in np.unique(tau_r):
        m = tau_r == t
        n = int(m.sum())
        if n > 0:
            w[m] = 1.0 / np.sqrt(n)
    return w


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


def _short_long_rmse(tau: np.ndarray, err: np.ndarray):
    med = float(np.median(tau))
    short = err[tau <= med]
    long = err[tau > med]
    rs = float(np.sqrt(np.mean(short**2))) if len(short) else float("nan")
    rl = float(np.sqrt(np.mean(long**2))) if len(long) else float("nan")
    return rs, rl


def _hit_bounds(x, lo, hi, names, tol=1e-8):
    hits = []
    for n, xi, a, b in zip(names, x, lo, hi):
        if abs(xi - a) <= tol or abs(xi - b) <= tol:
            hits.append(n)
    return hits


def fit_model(
    name,
    fn,
    names,
    x0,
    bounds,
    k,
    tau,
    iv,
    n_starts: int = 8,
    extra_starts: list | None = None,
) -> FitResult:
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    x0 = np.asarray(x0, float)
    rng = np.random.default_rng(0)
    w = _tenor_weights(tau)

    def resid(x):
        return w * (fn(k, tau, **_pack(names, x)) - iv)

    candidates = [x0]
    if extra_starts:
        for xs in extra_starts:
            candidates.append(np.clip(np.asarray(xs, float), lo + 1e-8, hi - 1e-8))
    for _ in range(max(0, n_starts - 1)):
        jitter = x0 + 0.30 * (hi - lo) * rng.normal(size=len(x0))
        candidates.append(np.clip(jitter, lo + 1e-8, hi - 1e-8))

    best = None
    for x_init in candidates:
        sol = least_squares(
            resid,
            x0=x_init,
            bounds=(lo, hi),
            max_nfev=1500,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        # Rank by unweighted RMSE so a bad local min cannot win on weights alone
        pred = fn(k, tau, **_pack(names, sol.x))
        cost = float(np.mean((pred - iv) ** 2))
        if best is None or cost < best[0]:
            best = (cost, sol)

    _, sol = best
    params = _pack(names, sol.x)
    pred = fn(k, tau, **params)
    err = pred - iv
    rs, rl = _short_long_rmse(tau, err)
    return FitResult(
        name=name,
        params=params,
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        rmse_weighted=float(np.sqrt(np.mean((w * err) ** 2))),
        rmse_short=rs,
        rmse_long=rl,
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
            ["v0", "v_inf", "kappa", "skew0", "smile0"],
            [v0, v0, 2.0, -0.5, 1.5],
            (
                [0.002, 0.002, 0.05, -4.0, 0.01],
                [0.80, 0.80, 25.0, 0.5, 12.0],
            ),
            k,
            tau,
            iv,
        ),
        fit_model(
            "1-factor Bergomi",
            bergomi1_like_iv,
            ["xi0", "kappa", "atm0", "skew0", "smile0"],
            [v0, 2.0, 0.05, -0.5, 1.5],
            (
                [0.002, 0.05, -0.30, -4.0, 0.01],
                [0.80, 25.0, 0.60, 0.5, 12.0],
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
            [v0, 10.0, 0.4, 0.08, 0.02, -0.7, -0.25, 2.0, 0.3],
            (
                [0.002, 1.5, 0.05, -0.30, -0.25, -5.0, -3.0, 0.0, 0.0],
                [0.80, 40.0, 4.0, 0.60, 0.50, 0.5, 0.5, 12.0, 4.0],
            ),
            k,
            tau,
            iv,
            n_starts=10,
            extra_starts=[
                # Fast smile + slow skew seeds (typical equity two-clock shape)
                [v0, 14.0, 0.5, 0.15, 0.05, -0.4, -0.35, 2.5, 0.4],
                [v0, 8.0, 1.5, 0.25, 0.08, -0.2, -0.45, 1.5, 0.6],
                [v0, 20.0, 2.0, 0.35, 0.10, 0.1, -0.50, 1.2, 0.8],
                [v0, 12.0, 0.8, 0.20, 0.04, -0.6, -0.25, 3.0, 0.3],
            ],
        ),
    ]


def two_factor_advantage(fits: list[FitResult]) -> dict:
    """
    Scalar that is large when 2F uniquely wins on short *and* long tenors.
    """
    by = {fr.name: fr for fr in fits}
    f2 = by["2-factor Bergomi"]
    rivals = [by["Heston-like"], by["1-factor Bergomi"]]
    best_rival_rmse = min(fr.rmse for fr in rivals)
    best_rival_short = min(fr.rmse_short for fr in rivals)
    best_rival_long = min(fr.rmse_long for fr in rivals)
    return {
        "rmse_ratio": float(best_rival_rmse / max(f2.rmse, 1e-12)),
        "short_ratio": float(best_rival_short / max(f2.rmse_short, 1e-12)),
        "long_ratio": float(best_rival_long / max(f2.rmse_long, 1e-12)),
        "joint_score": float(
            min(best_rival_short / max(f2.rmse_short, 1e-12),
                best_rival_long / max(f2.rmse_long, 1e-12))
        ),
    }


MODEL_FN = {
    "Black–Scholes": bs_iv,
    "Heston-like": heston_like_iv,
    "1-factor Bergomi": bergomi1_like_iv,
    "2-factor Bergomi": bergomi2_like_iv,
}
