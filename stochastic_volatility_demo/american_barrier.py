#!/usr/bin/env python3
"""
American down-and-out call via Longstaff–Schwartz MC.

No closed form under BS or stochastic volatility — the point of the demo.
Paths are simulated under BS / Heston / 1F Bergomi / 2F Bergomi dynamics
mapped from the calibrated IV-surface parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BarrierQuote:
    name: str
    price: float
    stderr: float
    european_barrier: float
    european_vanilla: float
    knock_prob: float
    early_exercise_premium: float
    params_used: dict
    n_paths: int
    n_steps: int


def _lsmc_american_dao(
    S: np.ndarray,
    alive: np.ndarray,
    K: float,
    r: float,
    dt: float,
    poly_deg: int = 2,
) -> tuple[np.ndarray, float]:
    """
    Longstaff–Schwartz on discrete monitoring grid.

    Parameters
    ----------
    S : (n_steps+1, n_paths) spot paths
    alive : (n_steps+1, n_paths) True if not yet knocked out
    """
    n_steps, n_paths = S.shape[0] - 1, S.shape[1]
    df = np.exp(-r * dt)
    # Cashflow at maturity for surviving paths
    payoff = np.where(alive[-1], np.maximum(S[-1] - K, 0.0), 0.0)
    cash = payoff.copy()
    exercise_count = 0

    for t in range(n_steps - 1, 0, -1):
        # Eligible: alive and in-the-money
        itm = alive[t] & (S[t] > K)
        if not np.any(itm):
            cash *= df
            continue
        idx = np.where(itm)[0]
        X = S[t, idx]
        Y = cash[idx] * df  # discounted continuation target
        # Polynomial basis regression
        A = np.vander(X, N=poly_deg + 1, increasing=True)
        try:
            beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
            cont = A @ beta
        except np.linalg.LinAlgError:
            cont = np.full_like(Y, Y.mean())
        intrinsic = X - K
        exercise = intrinsic > cont
        # Update cashflows
        ex_idx = idx[exercise]
        keep_idx = idx[~exercise]
        cash[ex_idx] = intrinsic[exercise]
        cash[keep_idx] *= df
        # Paths not ITM / not alive: just discount
        other = ~itm
        cash[other] *= df
        exercise_count += int(exercise.sum())

    # t=0: no exercise decision in LSMC (price is mean of discounted cash)
    price_paths = cash * df
    return price_paths, float(exercise_count) / max(n_paths, 1)


def _european_from_paths(S: np.ndarray, alive: np.ndarray, K: float, r: float, T: float):
    payoff = np.where(alive[-1], np.maximum(S[-1] - K, 0.0), 0.0)
    disc = np.exp(-r * T) * payoff
    return float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(len(disc)))


def _vanilla_european(S_T: np.ndarray, K: float, r: float, T: float):
    payoff = np.maximum(S_T - K, 0.0)
    disc = np.exp(-r * T) * payoff
    return float(disc.mean())


def _monitor_alive(S: np.ndarray, B: float) -> np.ndarray:
    """alive[t,i] True if path i has not hit barrier through time t."""
    hit = S <= B
    # cumulative any-hit
    cum = np.logical_or.accumulate(hit, axis=0)
    return ~cum


# ---------------------------------------------------------------------------
# Path simulators → (S grid, terminal extras)
# ---------------------------------------------------------------------------


def simulate_bs(S0, r, q, sigma, T, n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    S = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    S[0] = S0
    log_S = np.full(n_paths, np.log(S0))
    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        log_S = log_S + (r - q - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z
        S[t + 1] = np.exp(log_S)
    return S


def simulate_heston(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps, seed):
    """Full-truncation Euler for variance (demo-grade)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho = float(np.clip(rho, -0.999, 0.999))
    rho_perp = np.sqrt(max(1.0 - rho * rho, 0.0))
    S = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    S[0] = S0
    log_S = np.full(n_paths, np.log(S0))
    v = np.full(n_paths, max(v0, 1e-8))
    for t in range(n_steps):
        z_v = rng.standard_normal(n_paths)
        z_s = rho * z_v + rho_perp * rng.standard_normal(n_paths)
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * z_v
        v = np.maximum(v, 0.0)
        log_S = log_S + (r - q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z_s
        S[t + 1] = np.exp(log_S)
    return S


def simulate_bergomi1(
    S0, r, q, xi0, omega, kappa, rho, T, n_paths, n_steps, seed, stationary=True
):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho = float(np.clip(rho, -0.999, 0.999))
    rho_perp = np.sqrt(max(1.0 - rho * rho, 0.0))
    exp_k = np.exp(-kappa * dt)
    ou_std = (
        np.sqrt(max((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa), 0.0))
        if kappa > 1e-12
        else sqrt_dt
    )
    S = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    S[0] = S0
    log_S = np.full(n_paths, np.log(S0))
    X = np.zeros(n_paths)
    t = 0.0
    for i in range(n_steps):
        if stationary:
            v = xi0 * np.exp(omega * X - 0.25 * omega * omega / max(kappa, 1e-8))
        else:
            v = xi0 * np.exp(
                omega * X
                - (omega * omega / (4.0 * max(kappa, 1e-8)))
                * (1.0 - np.exp(-2.0 * kappa * t))
            )
        v = np.maximum(v, 1e-12)
        z_x = rng.standard_normal(n_paths)
        z_s = rho * z_x + rho_perp * rng.standard_normal(n_paths)
        X = exp_k * X + ou_std * z_x
        log_S = log_S + (r - q - 0.5 * v) * dt + np.sqrt(v) * sqrt_dt * z_s
        t += dt
        S[i + 1] = np.exp(log_S)
    return S


def simulate_bergomi2(
    S0,
    r,
    q,
    xi0,
    omega1,
    omega2,
    kappa1,
    kappa2,
    rho1,
    rho2,
    rho12,
    T,
    n_paths,
    n_steps,
    seed,
    stationary=True,
):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho12 = float(np.clip(rho12, -0.999, 0.999))
    rho1 = float(np.clip(rho1, -0.999, 0.999))
    rho2 = float(np.clip(rho2, -0.999, 0.999))
    rho12_perp = np.sqrt(max(1.0 - rho12 * rho12, 0.0))
    denom = max(1.0 - rho12 * rho12, 1e-12)
    a1 = (rho1 - rho12 * rho2) / denom
    a2 = (rho2 - rho12 * rho1) / denom
    a_perp = np.sqrt(max(1.0 - a1 * a1 - a2 * a2 - 2.0 * a1 * a2 * rho12, 0.0))
    exp_k1 = np.exp(-kappa1 * dt)
    exp_k2 = np.exp(-kappa2 * dt)
    ou1 = (
        np.sqrt(max((1.0 - np.exp(-2.0 * kappa1 * dt)) / (2.0 * kappa1), 0.0))
        if kappa1 > 1e-12
        else sqrt_dt
    )
    ou2 = (
        np.sqrt(max((1.0 - np.exp(-2.0 * kappa2 * dt)) / (2.0 * kappa2), 0.0))
        if kappa2 > 1e-12
        else sqrt_dt
    )
    S = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    S[0] = S0
    log_S = np.full(n_paths, np.log(S0))
    X1 = np.zeros(n_paths)
    X2 = np.zeros(n_paths)
    t = 0.0
    for i in range(n_steps):
        # Flat-forward Bergomi variance (stationary approx)
        if stationary:
            var1 = 1.0 / (2.0 * max(kappa1, 1e-8))
            var2 = 1.0 / (2.0 * max(kappa2, 1e-8))
            cov = rho12 / (kappa1 + kappa2) if (kappa1 + kappa2) > 1e-12 else 0.0
        else:
            var1 = (1.0 - np.exp(-2.0 * kappa1 * t)) / (2.0 * max(kappa1, 1e-8))
            var2 = (1.0 - np.exp(-2.0 * kappa2 * t)) / (2.0 * max(kappa2, 1e-8))
            cov = (
                rho12
                * (1.0 - np.exp(-(kappa1 + kappa2) * t))
                / max(kappa1 + kappa2, 1e-8)
            )
        drift = (
            0.5 * omega1 * omega1 * var1
            + 0.5 * omega2 * omega2 * var2
            + omega1 * omega2 * cov
        )
        v = xi0 * np.exp(omega1 * X1 + omega2 * X2 - drift)
        v = np.maximum(v, 1e-12)
        z1 = rng.standard_normal(n_paths)
        z2 = rho12 * z1 + rho12_perp * rng.standard_normal(n_paths)
        z_s = a1 * z1 + a2 * z2 + a_perp * rng.standard_normal(n_paths)
        X1 = exp_k1 * X1 + ou1 * z1
        X2 = exp_k2 * X2 + ou2 * z2
        log_S = log_S + (r - q - 0.5 * v) * dt + np.sqrt(v) * sqrt_dt * z_s
        t += dt
        S[i + 1] = np.exp(log_S)
    return S


# ---------------------------------------------------------------------------
# Map calibrated IV-surface params → dynamics for MC
# ---------------------------------------------------------------------------


def dynamics_from_fit(fit_name: str, params: dict, spot: float) -> dict:
    """Turn parametric IV fit coefficients into risk-neutral dynamics knobs."""
    if fit_name == "Black–Scholes":
        return {"model": "bs", "sigma": float(params["sigma"])}

    if fit_name == "Heston-like":
        v0 = float(params["v0"])
        theta = float(params["v_inf"])
        kappa = float(params["kappa"])
        skew0 = float(params["skew0"])
        smile0 = float(params["smile0"])
        volvol = float(np.clip(0.5 + 0.35 * np.sqrt(max(smile0, 0.0)), 0.2, 2.5))
        rho = float(np.clip(skew0 / max(volvol, 0.3), -0.9, 0.9))
        return {
            "model": "heston",
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "xi": volvol,
            "rho": rho,
        }

    if fit_name == "1-factor Bergomi":
        xi0 = float(params["xi0"])
        kappa = float(params["kappa"])
        atm0 = float(params["atm0"])
        skew0 = float(params["skew0"])
        smile0 = float(params["smile0"])
        omega = float(
            np.clip(1.0 + 4.0 * abs(atm0) + 0.6 * np.sqrt(max(smile0, 0.0)), 0.4, 5.0)
        )
        rho = float(np.clip(skew0 / max(omega, 0.5), -0.9, 0.9))
        return {
            "model": "bergomi1",
            "xi0": xi0,
            "omega": omega,
            "kappa": kappa,
            "rho": rho,
        }

    if fit_name == "2-factor Bergomi":
        xi0 = float(params["xi0"])
        k1 = float(params["kappa1"])
        k2 = float(params["kappa2"])
        atm1 = float(params["atm1"])
        atm2 = float(params["atm2"])
        skew1 = float(params["skew1"])
        skew2 = float(params["skew2"])
        smile1 = float(params["smile1"])
        smile2 = float(params["smile2"])
        omega1 = float(
            np.clip(1.2 + 5.0 * abs(atm1) + 0.5 * np.sqrt(max(smile1, 0.0)), 0.4, 6.0)
        )
        omega2 = float(
            np.clip(0.6 + 4.0 * abs(atm2) + 0.5 * np.sqrt(max(smile2, 0.0)), 0.2, 4.0)
        )
        rho1 = float(np.clip(skew1 / max(omega1, 0.5), -0.9, 0.9))
        rho2 = float(np.clip(skew2 / max(omega2, 0.5), -0.9, 0.9))
        return {
            "model": "bergomi2",
            "xi0": xi0,
            "omega1": omega1,
            "omega2": omega2,
            "kappa1": k1,
            "kappa2": k2,
            "rho1": rho1,
            "rho2": rho2,
            "rho12": 0.2,
        }

    raise ValueError(f"Unknown fit name: {fit_name}")


def price_american_down_out_call(
    fit_name: str,
    params: dict,
    S0: float,
    K: float,
    B: float,
    T: float,
    r: float = 0.04,
    q: float = 0.005,
    n_paths: int = 20_000,
    n_steps: int = 48,
    seed: int = 0,
) -> BarrierQuote:
    """
    Price an American down-and-out call under dynamics implied by a surface fit.

    Knock-out if S_t ≤ B on the discrete grid; early exercise via LSMC.
    """
    if not (0.0 < B < S0):
        raise ValueError("Need 0 < B < S0 for a live down-and-out call")
    dyn = dynamics_from_fit(fit_name, params, S0)
    model = dyn["model"]

    if model == "bs":
        S = simulate_bs(S0, r, q, dyn["sigma"], T, n_paths, n_steps, seed)
    elif model == "heston":
        S = simulate_heston(
            S0,
            r,
            q,
            dyn["v0"],
            dyn["kappa"],
            dyn["theta"],
            dyn["xi"],
            dyn["rho"],
            T,
            n_paths,
            n_steps,
            seed,
        )
    elif model == "bergomi1":
        S = simulate_bergomi1(
            S0,
            r,
            q,
            dyn["xi0"],
            dyn["omega"],
            dyn["kappa"],
            dyn["rho"],
            T,
            n_paths,
            n_steps,
            seed,
        )
    elif model == "bergomi2":
        S = simulate_bergomi2(
            S0,
            r,
            q,
            dyn["xi0"],
            dyn["omega1"],
            dyn["omega2"],
            dyn["kappa1"],
            dyn["kappa2"],
            dyn["rho1"],
            dyn["rho2"],
            dyn["rho12"],
            T,
            n_paths,
            n_steps,
            seed,
        )
    else:
        raise ValueError(model)

    alive = _monitor_alive(S, B)
    knock_prob = float((~alive[-1]).mean())
    euro_b, _ = _european_from_paths(S, alive, K, r, T)
    euro_v = _vanilla_european(S[-1], K, r, T)

    dt = T / n_steps
    am_paths, _ = _lsmc_american_dao(S, alive, K, r, dt)
    price = float(am_paths.mean())
    stderr = float(am_paths.std(ddof=1) / np.sqrt(n_paths))
    eep = price - euro_b

    return BarrierQuote(
        name=fit_name,
        price=price,
        stderr=stderr,
        european_barrier=euro_b,
        european_vanilla=euro_v,
        knock_prob=knock_prob,
        early_exercise_premium=eep,
        params_used=dyn,
        n_paths=n_paths,
        n_steps=n_steps,
    )


def price_barrier_grid(
    fit_name: str,
    params: dict,
    S0: float,
    K: float,
    barriers: np.ndarray,
    T: float,
    r: float,
    q: float,
    n_paths: int = 12_000,
    n_steps: int = 40,
    seed: int = 0,
) -> list[BarrierQuote]:
    return [
        price_american_down_out_call(
            fit_name,
            params,
            S0,
            K,
            float(B),
            T,
            r=r,
            q=q,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed + i,
        )
        for i, B in enumerate(barriers)
    ]


@dataclass
class BarrierGreeks:
    name: str
    S: np.ndarray
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    vega: np.ndarray
    spot0: float
    delta0: float
    gamma0: float
    theta0: float
    vega0: float
    price0: float


def _bump_vol_params(fit_name: str, params: dict, bump: float) -> dict:
    """Relative bump on the main volatility amplitude parameter(s)."""
    p = dict(params)
    if fit_name == "Black–Scholes":
        p["sigma"] = float(params["sigma"]) * (1.0 + bump)
    elif fit_name == "Heston-like":
        p["v0"] = float(params["v0"]) * (1.0 + bump)
        p["v_inf"] = float(params["v_inf"]) * (1.0 + bump)
    elif fit_name == "1-factor Bergomi":
        p["xi0"] = float(params["xi0"]) * (1.0 + bump)
    elif fit_name == "2-factor Bergomi":
        p["xi0"] = float(params["xi0"]) * (1.0 + bump)
    return p


def compute_american_barrier_greeks(
    fit_name: str,
    params: dict,
    S0: float,
    K: float,
    B: float,
    T: float,
    r: float = 0.04,
    q: float = 0.005,
    spot_grid: np.ndarray | None = None,
    n_paths: int = 8_000,
    n_steps: int = 40,
    seed: int = 7,
    dS_rel: float = 0.01,
    dT: float = 1.0 / 365.0,
    dVol: float = 0.05,
) -> BarrierGreeks:
    """
    Finite-difference Greeks for the American DAO call under one model.

    Delta / Gamma along a spot grid (barrier must stay below each spot);
    Theta via maturity bump; Vega via relative bump on the main vol amplitude.
    """
    if spot_grid is None:
        # Stay safely above the barrier
        lo = max(B * 1.02, S0 * 0.92)
        hi = S0 * 1.12
        spot_grid = np.linspace(lo, hi, 11)

    def _price(S, T_, pars, seed_):
        # Keep barrier fixed in absolute terms; skip spots that would start KO'd
        if S <= B:
            return 0.0
        return price_american_down_out_call(
            fit_name,
            pars,
            S0=float(S),
            K=K,
            B=B,
            T=float(T_),
            r=r,
            q=q,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed_,
        ).price

    prices = []
    deltas = []
    gammas = []
    # Common seed across spot/vol/time bumps → correlated MC noise, stabler FD
    for i, S in enumerate(spot_grid):
        h = max(dS_rel * S, 1e-4 * S0)
        # Ensure S-h still above barrier for central differences
        if S - h <= B:
            h = 0.5 * (S - B)
        s_i = seed + 17 * i
        up = _price(S + h, T, params, s_i)
        mid = _price(S, T, params, s_i)
        dn = _price(S - h, T, params, s_i)
        prices.append(mid)
        deltas.append((up - dn) / (2.0 * h))
        gammas.append((up - 2.0 * mid + dn) / (h * h))

    # Spot-centered Theta / Vega at S0
    h0 = max(dS_rel * S0, 1e-4 * S0)
    if S0 - h0 <= B:
        h0 = 0.5 * (S0 - B)
    s0 = seed
    p0 = _price(S0, T, params, s0)
    p_up = _price(S0 + h0, T, params, s0)
    p_dn = _price(S0 - h0, T, params, s0)
    delta0 = (p_up - p_dn) / (2.0 * h0)
    gamma0 = (p_up - 2.0 * p0 + p_dn) / (h0 * h0)

    T_left = max(T - dT, 1e-4)
    p_tm = _price(S0, T_left, params, s0)
    # calendar Theta ≈ ∂V/∂t with τ = T-t decreasing → (V(τ-dτ)-V(τ))/dτ
    theta0 = (p_tm - p0) / dT

    p_vol = _price(S0, T, _bump_vol_params(fit_name, params, dVol), s0)
    # Vega as dV / d(relative vol bump)
    vega0 = (p_vol - p0) / dVol

    thetas = []
    vegas = []
    for i, S in enumerate(spot_grid):
        mid = prices[i]
        s_i = seed + 17 * i
        p_tm = _price(S, T_left, params, s_i)
        thetas.append((p_tm - mid) / dT)
        p_vol = _price(S, T, _bump_vol_params(fit_name, params, dVol), s_i)
        vegas.append((p_vol - mid) / dVol)

    return BarrierGreeks(
        name=fit_name,
        S=np.asarray(spot_grid, float),
        price=np.asarray(prices, float),
        delta=np.asarray(deltas, float),
        gamma=np.asarray(gammas, float),
        theta=np.asarray(thetas, float),
        vega=np.asarray(vegas, float),
        spot0=float(S0),
        delta0=float(delta0),
        gamma0=float(gamma0),
        theta0=float(theta0),
        vega0=float(vega0),
        price0=float(p0),
    )
