import numpy as np


def Heston_Monte_Carlo(
    S0,
    K,
    T,
    r,
    q,
    v0,
    kappa,
    theta,
    sigma,
    rho,
    call_put="Call",
    n_paths=200_000,
    n_steps=200,
    psi_c=1.5,
    gamma1=0.5,
    gamma2=0.5,
    seed=None,
    return_stderr=False,
):
    """
    Monte Carlo pricer for a European option under the Heston model
    using Andersen's Quadratic-Exponential (QE) scheme.

    Model:
        dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
        dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW2_t
        corr(dW1, dW2) = rho

    Parameters
    ----------
    S0 : float
        Initial spot.
    K : float
        Strike.
    T : float
        Maturity in years.
    r : float
        Risk-free rate.
    q : float
        Dividend yield / foreign rate.
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run variance.
    sigma : float
        Vol of vol.
    rho : float
        Correlation between asset and variance Brownian motions.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps.
    psi_c : float
        QE switching threshold, commonly 1.5.
    gamma1, gamma2 : float
        Weights in Andersen's log-stock update, often 0.5 / 0.5.
    seed : int or None
        RNG seed.
    return_stderr : bool
        If True, also returns Monte Carlo standard error.

    Returns
    -------
    price : float
        Discounted MC price.
    stderr : float, optional
        Standard error of the discounted payoff estimator.
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    exp_kdt = np.exp(-kappa * dt)

    # Precompute constants for variance moments
    K0 = -(rho * kappa * theta / sigma) * dt
    K1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = gamma1 * dt * (1.0 - rho * rho)
    K4 = gamma2 * dt * (1.0 - rho * rho)

    x = np.full(n_paths, np.log(S0), dtype=np.float64)
    v = np.full(n_paths, v0, dtype=np.float64)

    for _ in range(n_steps):
        # Conditional moments of v_{t+dt} | v_t
        m = theta + (v - theta) * exp_kdt
        s2 = (
            v * sigma * sigma * exp_kdt * (1.0 - exp_kdt) / kappa
            + theta * sigma * sigma * (1.0 - exp_kdt) ** 2 / (2.0 * kappa)
        )
        psi = s2 / (m * m)

        z_v = rng.standard_normal(n_paths)
        u = rng.random(n_paths)

        v_next = np.empty_like(v)

        # Case 1: psi <= psi_c, quadratic Gaussian transform
        idx1 = psi <= psi_c
        psi1 = psi[idx1]
        m1 = m[idx1]
        z1 = z_v[idx1]

        if psi1.size > 0:
            b2 = 2.0 / psi1 - 1.0 + np.sqrt(2.0 / psi1) * np.sqrt(2.0 / psi1 - 1.0)
            a = m1 / (1.0 + b2)
            b = np.sqrt(b2)
            v_next[idx1] = a * (b + z1) ** 2

        # Case 2: psi > psi_c, point-mass + exponential tail
        idx2 = ~idx1
        psi2 = psi[idx2]
        m2 = m[idx2]
        u2 = u[idx2]

        if psi2.size > 0:
            p = (psi2 - 1.0) / (psi2 + 1.0)
            beta = (1.0 - p) / m2

            # Inverse CDF sampling:
            # V = 0 with prob p, else exponential tail
            v_tmp = np.zeros_like(m2)
            nonzero = u2 > p
            v_tmp[nonzero] = -np.log((1.0 - u2[nonzero]) / (1.0 - p[nonzero])) / beta[nonzero]
            v_next[idx2] = v_tmp

        # Independent normal for stock update
        z_x = rng.standard_normal(n_paths)

        # Safeguard against tiny negative values from roundoff
        v_clip = np.maximum(v, 0.0)
        v_next_clip = np.maximum(v_next, 0.0)

        drift = (r - q) * dt + K0 + K1 * v_clip + K2 * v_next_clip
        var_term = np.maximum(K3 * v_clip + K4 * v_next_clip, 0.0)

        x = x + drift + np.sqrt(var_term) * z_x
        v = v_next_clip

    ST = np.exp(x)
    if call_put.lower() == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc_payoff = np.exp(-r * T) * payoff

    price = disc_payoff.mean()

    if return_stderr:
        stderr = disc_payoff.std(ddof=1) / np.sqrt(n_paths)
        return price, stderr

    return price


# ============================================================
# 1-factor Bergomi Monte Carlo
# ============================================================


def _v_bergomi_np(X, xi0, omega, kappa, t=None, stationary=True, eps=1e-12):
    """Numpy instantaneous variance under 1-factor Bergomi (flat xi0)."""
    X = np.asarray(X, dtype=np.float64)
    xi0 = np.maximum(float(xi0), eps)
    omega = float(omega)
    kappa = max(float(kappa), eps)

    if stationary or t is None:
        expo = omega * X - (omega ** 2) / (4.0 * kappa)
    else:
        t = np.asarray(t, dtype=np.float64)
        expo = omega * X - (omega ** 2) / (4.0 * kappa) * (
            1.0 - np.exp(-2.0 * kappa * t)
        )

    return np.maximum(xi0 * np.exp(expo), eps)


def Bergomi_Monte_Carlo(
    S0,
    K,
    T,
    r,
    q,
    xi0,
    omega,
    kappa,
    rho,
    X0=0.0,
    call_put="Call",
    n_paths=200_000,
    n_steps=200,
    stationary=True,
    seed=None,
    return_stderr=False,
    return_paths=False,
):
    """
    Monte Carlo pricer for a European option under the 1-factor Bergomi model.

    Dynamics (risk-neutral):
        dS_t / S_t = (r - q) dt + sqrt(v_t) dW^S_t
        dX_t       = -kappa X_t dt + dW^X_t
        d<W^S, W^X>_t = rho dt

    Instantaneous variance with flat initial forward variance xi0:

        v(t, X) = xi0 * exp( omega*X - (omega^2/(4*kappa))*(1 - exp(-2*kappa*t)) )

    or the stationary approximation (``stationary=True``, default):

        v(X) = xi0 * exp( omega*X - omega^2/(4*kappa) )

    Simulation scheme
    -----------------
    - Exact Gaussian transition for the OU factor X.
    - Log-Euler step for S with Brownian correlated to the OU innovation
      (same correlation structure as the pricing PDE).

    Parameters
    ----------
    S0, K, T, r, q : float
        Spot, strike, maturity, rate, dividend yield.
    xi0 : float
        Initial flat forward variance.
    omega : float
        Vol-of-variance (Bergomi).
    kappa : float
        Mean-reversion speed of X.
    rho : float
        Spot–factor correlation in (-1, 1).
    X0 : float
        Initial OU factor (typically 0).
    call_put : {"Call", "Put"}
    n_paths, n_steps : int
    stationary : bool
        Use stationary variance formula if True.
    seed : int or None
    return_stderr : bool
        Also return MC standard error.
    return_paths : bool
        If True, also return terminal spot and factor arrays.

    Returns
    -------
    price : float
    stderr : float, optional
    (ST, XT) : tuple of ndarray, optional
        Only if ``return_paths=True`` (appended after price / stderr).
    """
    cp = call_put.lower()
    if cp not in {"call", "put"}:
        raise ValueError("call_put must be either 'Call' or 'Put'")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must lie strictly inside (-1, 1)")
    if T <= 0.0:
        intrinsic = max(S0 - K, 0.0) if cp == "call" else max(K - S0, 0.0)
        if return_stderr and return_paths:
            return float(intrinsic), 0.0, (np.full(n_paths, S0), np.full(n_paths, X0))
        if return_stderr:
            return float(intrinsic), 0.0
        if return_paths:
            return float(intrinsic), (np.full(n_paths, S0), np.full(n_paths, X0))
        return float(intrinsic)

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    exp_kdt = np.exp(-kappa * dt)

    # Exact conditional std of OU increment:
    # X_{t+dt} = e^{-kappa dt} X_t + sqrt( (1 - e^{-2 kappa dt}) / (2 kappa) ) Z
    if kappa > 1e-12:
        ou_std = np.sqrt(max((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa), 0.0))
    else:
        ou_std = sqrt_dt

    rho_perp = np.sqrt(max(1.0 - rho * rho, 0.0))

    log_S = np.full(n_paths, np.log(S0), dtype=np.float64)
    X = np.full(n_paths, float(X0), dtype=np.float64)
    t = 0.0

    for _ in range(n_steps):
        # Variance at the left endpoint of the step
        v = _v_bergomi_np(
            X, xi0=xi0, omega=omega, kappa=kappa, t=t, stationary=stationary
        )
        sqrt_v = np.sqrt(v)

        z_X = rng.standard_normal(n_paths)
        z_perp = rng.standard_normal(n_paths)
        z_S = rho * z_X + rho_perp * z_perp

        # Exact OU step for the factor
        X = exp_kdt * X + ou_std * z_X

        # Log-Euler step for the spot
        log_S = log_S + (r - q - 0.5 * v) * dt + sqrt_v * sqrt_dt * z_S
        t += dt

    ST = np.exp(log_S)
    if cp == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc_payoff = np.exp(-r * T) * payoff
    price = float(disc_payoff.mean())

    out = [price]
    if return_stderr:
        stderr = float(disc_payoff.std(ddof=1) / np.sqrt(n_paths))
        out.append(stderr)
    if return_paths:
        out.append((ST, X))

    if len(out) == 1:
        return out[0]
    return tuple(out)


def bergomi_mc_grid(
    S_grid,
    tau_grid,
    K,
    r,
    q,
    xi0,
    omega,
    kappa,
    rho,
    X0=0.0,
    call_put="Call",
    n_paths=50_000,
    n_steps=100,
    stationary=True,
    seed=42,
):
    """
    Price a European option on a (S, tau) grid via Bergomi Monte Carlo.

    Returns
    -------
    prices : ndarray, shape (len(S_grid), len(tau_grid))
    stderrs : ndarray, same shape
    """
    S_grid = np.asarray(S_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    prices = np.empty((S_grid.size, tau_grid.size), dtype=np.float64)
    stderrs = np.empty_like(prices)

    # Independent seeds per cell for reproducibility without shared RNG state
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(S_grid.size * tau_grid.size)
    k = 0
    for i, S0 in enumerate(S_grid):
        for j, tau in enumerate(tau_grid):
            # Scale steps roughly with maturity
            steps = max(
                int(round(n_steps * max(tau, 1e-6) / max(float(tau_grid.max()), 1e-6))),
                10,
            )
            p, se = Bergomi_Monte_Carlo(
                S0=float(S0),
                K=K,
                T=float(tau),
                r=r,
                q=q,
                xi0=xi0,
                omega=omega,
                kappa=kappa,
                rho=rho,
                X0=X0,
                call_put=call_put,
                n_paths=n_paths,
                n_steps=steps,
                stationary=stationary,
                seed=child_seeds[k],
                return_stderr=True,
            )
            prices[i, j] = p
            stderrs[i, j] = se
            k += 1

    return prices, stderrs
