#!/usr/bin/env python3
"""
Parametric IV-surface models and least-squares fits to market quotes.

These are *stylized* closed-form IV parameterizations (not the PINN pricers).
They are sized to show which *factor structure* can absorb a real equity
surface: flat / one clock / two clocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def bs_iv(k, tau, sigma=0.2):
    return np.full_like(k, float(sigma), dtype=float)


def heston_like_iv(k, tau, v0=0.04, rho=-0.65, volvol=0.55, kappa=2.0):
    atm = np.sqrt(max(v0, 1e-8)) * (1.0 + 0.04 * np.exp(-kappa * tau))
    skew = rho * volvol * 0.32 * np.exp(-0.55 * tau) * k
    smile = 0.10 * (k**2) * np.exp(-0.35 * tau)
    return np.clip(atm + skew + smile, 0.05, 0.95)


def bergomi1_like_iv(k, tau, xi0=0.04, omega=1.2, rho=-0.35, kappa=2.0):
    atm = np.sqrt(max(xi0, 1e-8)) * (1.0 + 0.10 * omega * np.exp(-kappa * tau))
    skew = rho * omega * 0.22 * np.exp(-0.7 * kappa * tau) * k
    smile = 0.28 * omega * (k**2) * np.exp(-1.4 * tau)
    return np.clip(atm + skew + smile, 0.05, 1.05)


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
    w_fast = np.exp(-kappa1 * tau)
    w_slow = np.exp(-kappa2 * tau)
    atm = np.sqrt(max(xi0, 1e-8)) * (
        1.0 + 0.12 * omega1 * w_fast + 0.10 * omega2 * w_slow
    )
    skew = (rho1 * omega1 * 0.18 * w_fast + rho2 * omega2 * 0.20 * w_slow) * k
    smile = 0.38 * omega1 * (k**2) * w_fast + 0.12 * omega2 * (k**2) * w_slow
    return np.clip(atm + skew + smile, 0.05, 1.15)


@dataclass
class FitResult:
    name: str
    params: dict
    rmse: float
    mae: float
    rmse_by_tenor: dict


def _pack(names, x):
    return {n: float(v) for n, v in zip(names, x)}


def fit_model(name, fn, names, x0, bounds, k, tau, iv) -> FitResult:
    lo, hi = bounds

    def resid(x):
        kw = _pack(names, x)
        return fn(k, tau, **kw) - iv

    sol = least_squares(resid, x0=np.asarray(x0, float), bounds=(lo, hi), max_nfev=400)
    pred = fn(k, tau, **_pack(names, sol.x))
    err = pred - iv
    tenors = np.unique(np.round(tau, 6))
    by = {}
    for t in tenors:
        m = np.isclose(tau, t, atol=1e-4)
        if m.any():
            by[float(t)] = float(np.sqrt(np.mean(err[m] ** 2)))
    return FitResult(
        name=name,
        params=_pack(names, sol.x),
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        rmse_by_tenor=by,
    )


def fit_all(k, tau, iv) -> list[FitResult]:
    """Fit BS / Heston / Bergomi-1F / Bergomi-2F parameterizations."""
    atm0 = float(np.median(iv[np.abs(k) < 0.03])) if np.any(np.abs(k) < 0.03) else float(
        np.median(iv)
    )
    fits = [
        fit_model(
            "Black–Scholes",
            bs_iv,
            ["sigma"],
            [atm0],
            ([0.05], [0.8]),
            k,
            tau,
            iv,
        ),
        fit_model(
            "Heston-like",
            heston_like_iv,
            ["v0", "rho", "volvol", "kappa"],
            [atm0**2, -0.6, 0.5, 2.0],
            ([0.005, -0.95, 0.05, 0.3], [0.25, 0.0, 2.0, 8.0]),
            k,
            tau,
            iv,
        ),
        fit_model(
            "1-factor Bergomi",
            bergomi1_like_iv,
            ["xi0", "omega", "rho", "kappa"],
            [atm0**2, 1.0, -0.35, 2.0],
            ([0.005, 0.1, -0.9, 0.3], [0.25, 3.0, 0.2, 10.0]),
            k,
            tau,
            iv,
        ),
        fit_model(
            "2-factor Bergomi",
            bergomi2_like_iv,
            ["xi0", "omega1", "omega2", "kappa1", "kappa2", "rho1", "rho2"],
            [atm0**2, 1.4, 0.6, 8.0, 0.5, -0.25, -0.15],
            (
                [0.005, 0.2, 0.05, 2.0, 0.1, -0.9, -0.7],
                [0.25, 3.0, 2.0, 20.0, 2.0, 0.2, 0.2],
            ),
            k,
            tau,
            iv,
        ),
    ]
    return fits


MODEL_FN = {
    "Black–Scholes": bs_iv,
    "Heston-like": heston_like_iv,
    "1-factor Bergomi": bergomi1_like_iv,
    "2-factor Bergomi": bergomi2_like_iv,
}
