import numpy as np

try:
    from scipy.stats import norm as _scipy_norm
    from scipy.stats import qmc as _scipy_qmc
except ImportError:  # pragma: no cover
    _scipy_norm = None
    _scipy_qmc = None


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


def _v_bergomi_np(X, xi0, omega, kappa, t=None, stationary=True, eps=1e-12, v_max=None):
    """Numpy instantaneous variance under 1-factor Bergomi (flat xi0)."""
    X = np.asarray(X, dtype=np.float64)
    xi0 = np.maximum(float(xi0), eps)
    omega = float(omega)
    kappa = max(float(kappa), max(eps, 1e-3))

    if stationary or t is None:
        expo = omega * X - (omega ** 2) / (4.0 * kappa)
    else:
        t = np.asarray(t, dtype=np.float64)
        expo = omega * X - (omega ** 2) / (4.0 * kappa) * (
            1.0 - np.exp(-2.0 * kappa * t)
        )

    expo = np.clip(expo, -20.0, 20.0)
    v = np.maximum(xi0 * np.exp(expo), eps)
    if v_max is not None:
        v = np.minimum(v, float(v_max))
    return v


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


# ============================================================
# Bergomi QMC + Antithetic Differencing (ADD)
# ============================================================


def _next_power_of_two(n: int) -> int:
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


def _sobol_standard_normals(n_paths: int, dim: int, seed: int | None = None):
    """Scrambled Sobol' points mapped to N(0,1) via inverse CDF."""
    if _scipy_qmc is None or _scipy_norm is None:
        raise ImportError("scipy is required for Bergomi QMC+ADD pricing")

    n_pow2 = _next_power_of_two(n_paths)
    engine = _scipy_qmc.Sobol(d=int(dim), scramble=True, seed=seed)
    # skip first point of Sobol' engines for better uniformity in low dims
    try:
        engine.fast_forward(1)
    except Exception:
        pass
    u = engine.random(n_pow2)[:n_paths]
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return _scipy_norm.ppf(u).astype(np.float64)


def _bergomi_simulate_from_normals(
    S0,
    T,
    r,
    q,
    xi0,
    omega,
    kappa,
    rho,
    z_X,
    z_perp,
    X0=0.0,
    stationary=True,
):
    """
    Simulate Bergomi paths given unit normals of shape (n_paths, n_steps).

    Returns
    -------
    ST : ndarray (n_paths,)
    XT : ndarray (n_paths,)
    mult : ndarray (n_paths,)
        Terminal multiplicative factor ST / S0 (independent of S0).
    """
    n_paths, n_steps = z_X.shape
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    exp_kdt = np.exp(-kappa * dt)
    if kappa > 1e-12:
        ou_std = np.sqrt(max((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa), 0.0))
    else:
        ou_std = sqrt_dt
    rho_perp = np.sqrt(max(1.0 - rho * rho, 0.0))

    log_S = np.full(n_paths, np.log(S0), dtype=np.float64)
    X = np.full(n_paths, float(X0), dtype=np.float64)
    t = 0.0

    for k in range(n_steps):
        v = _v_bergomi_np(
            X, xi0=xi0, omega=omega, kappa=kappa, t=t, stationary=stationary
        )
        sqrt_v = np.sqrt(v)
        zS = rho * z_X[:, k] + rho_perp * z_perp[:, k]
        X = exp_kdt * X + ou_std * z_X[:, k]
        log_S = log_S + (r - q - 0.5 * v) * dt + sqrt_v * sqrt_dt * zS
        t += dt

    ST = np.exp(log_S)
    mult = ST / float(S0)
    return ST, X, mult


def Bergomi_QMC_ADD(
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
    n_paths=131_072,
    n_steps=128,
    stationary=True,
    seed=42,
    return_stderr=False,
    return_paths=False,
    return_mult=False,
):
    """
    Bergomi European pricer using scrambled Sobol' QMC with antithetic
    differencing (ADD): each low-discrepancy path is paired with its
    sign-flipped antithetic counterpart, and the estimator averages the pair.

    ``n_paths`` is the *total* number of simulated trajectories after ADD
    (so ``n_paths // 2`` Sobol' base points). Prefer powers of two.
    """
    cp = call_put.lower()
    if cp not in {"call", "put"}:
        raise ValueError("call_put must be either 'Call' or 'Put'")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must lie strictly inside (-1, 1)")

    n_paths = int(n_paths)
    if n_paths % 2 != 0:
        n_paths += 1
    n_base = n_paths // 2
    n_steps = int(n_steps)

    if T <= 0.0:
        intrinsic = max(S0 - K, 0.0) if cp == "call" else max(K - S0, 0.0)
        mult = np.ones(n_paths, dtype=np.float64)
        ST = np.full(n_paths, S0, dtype=np.float64)
        XT = np.full(n_paths, X0, dtype=np.float64)
        out = [float(intrinsic)]
        if return_stderr:
            out.append(0.0)
        if return_paths:
            out.append((ST, XT))
        if return_mult:
            out.append(mult)
        return out[0] if len(out) == 1 else tuple(out)

    # Two independent Gaussians per time step
    z = _sobol_standard_normals(n_base, dim=2 * n_steps, seed=seed)
    z = z.reshape(n_base, n_steps, 2)
    # Antithetic differencing (ADD): append -Z
    z = np.concatenate([z, -z], axis=0)
    z_X = z[:, :, 0]
    z_perp = z[:, :, 1]

    ST, XT, mult = _bergomi_simulate_from_normals(
        S0=S0,
        T=T,
        r=r,
        q=q,
        xi0=xi0,
        omega=omega,
        kappa=kappa,
        rho=rho,
        z_X=z_X,
        z_perp=z_perp,
        X0=X0,
        stationary=stationary,
    )

    if cp == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc_payoff = np.exp(-r * T) * payoff
    # ADD pair averages for a lower-variance unbiased estimator
    pair_avg = 0.5 * (disc_payoff[:n_base] + disc_payoff[n_base:])
    price = float(pair_avg.mean())

    out = [price]
    if return_stderr:
        # RQMC stderr from independent antithetic-pair averages
        stderr = float(pair_avg.std(ddof=1) / np.sqrt(n_base)) if n_base > 1 else 0.0
        out.append(stderr)
    if return_paths:
        out.append((ST, XT))
    if return_mult:
        out.append(mult)
    return out[0] if len(out) == 1 else tuple(out)


def bergomi_qmc_add_grid(
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
    n_paths=131_072,
    n_steps=128,
    stationary=True,
    seed=42,
):
    """
    Price on a (S, tau) grid with one QMC+ADD noise field per maturity.

    Because the terminal multiplier ST/S0 is independent of S0 under the
    log-Euler scheme, all spots share the same simulated paths for each tau.
    """
    S_grid = np.asarray(S_grid, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    prices = np.empty((S_grid.size, tau_grid.size), dtype=np.float64)
    stderrs = np.empty_like(prices)

    cp = call_put.lower()
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(tau_grid.size)

    for j, tau in enumerate(tau_grid):
        steps = max(
            int(round(n_steps * max(float(tau), 1e-6) / max(float(tau_grid.max()), 1e-6))),
            16,
        )
        # Simulate with S0=1 to obtain multiplicative factors
        _price0, se0, mult = Bergomi_QMC_ADD(
            S0=1.0,
            K=K,  # unused for mult; overwritten below
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
            seed=int(child_seeds[j].generate_state(1)[0]),
            return_stderr=True,
            return_mult=True,
        )
        n_base = mult.size // 2
        disc = np.exp(-r * float(tau))
        for i, S0 in enumerate(S_grid):
            ST = float(S0) * mult
            if cp == "call":
                payoff = np.maximum(ST - K, 0.0)
            else:
                payoff = np.maximum(K - ST, 0.0)
            disc_payoff = disc * payoff
            pair_avg = 0.5 * (disc_payoff[:n_base] + disc_payoff[n_base:])
            prices[i, j] = float(pair_avg.mean())
            stderrs[i, j] = (
                float(pair_avg.std(ddof=1) / np.sqrt(n_base)) if n_base > 1 else 0.0
            )
        # silence unused
        _ = (_price0, se0)

    return prices, stderrs


def Bergomi_QMC_ADD_Greeks(
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
    n_paths=131_072,
    n_steps=128,
    stationary=True,
    seed=42,
    bump_S=1e-4,
    bump_T=1e-4,
):
    """
    Bergomi price / Delta / Theta via QMC+ADD.

    Delta uses the pathwise estimator (same QMC+ADD paths).
    Theta uses central finite differences in T with common random numbers
    (shared unit normals, maturity-dependent time step).
    """
    cp = call_put.lower()
    n_paths = int(n_paths)
    if n_paths % 2 != 0:
        n_paths += 1
    n_base = n_paths // 2
    n_steps = int(n_steps)

    if T <= bump_T:
        # fall back: intrinsic delta, zero theta near expiry
        if cp == "call":
            price = max(S0 - K, 0.0)
            delta = 1.0 if S0 > K else 0.0
        else:
            price = max(K - S0, 0.0)
            delta = -1.0 if S0 < K else 0.0
        return float(price), float(delta), 0.0

    z = _sobol_standard_normals(n_base, dim=2 * n_steps, seed=seed)
    z = z.reshape(n_base, n_steps, 2)
    z = np.concatenate([z, -z], axis=0)
    z_X = z[:, :, 0]
    z_perp = z[:, :, 1]

    def _price_delta_at(T_loc):
        ST, _, mult = _bergomi_simulate_from_normals(
            S0=S0,
            T=T_loc,
            r=r,
            q=q,
            xi0=xi0,
            omega=omega,
            kappa=kappa,
            rho=rho,
            z_X=z_X,
            z_perp=z_perp,
            X0=X0,
            stationary=stationary,
        )
        disc = np.exp(-r * T_loc)
        if cp == "call":
            payoff = np.maximum(ST - K, 0.0)
            # pathwise delta: disc * 1_{ST>K} * (ST/S0)
            pw = disc * (ST > K) * mult
        else:
            payoff = np.maximum(K - ST, 0.0)
            pw = -disc * (ST < K) * mult
        disc_payoff = disc * payoff
        pair_p = 0.5 * (disc_payoff[:n_base] + disc_payoff[n_base:])
        pair_d = 0.5 * (pw[:n_base] + pw[n_base:])
        return float(pair_p.mean()), float(pair_d.mean())

    price, delta = _price_delta_at(T)

    # Common-random-number central difference for calendar Theta = dV/dT
    # (PINN reports ∂V/∂t = -∂V/∂tau; with fixed calendar issuance this
    #  matches -dV/dT for European claims.)
    p_up, _ = _price_delta_at(T + bump_T)
    p_dn, _ = _price_delta_at(T - bump_T)
    dV_dT = (p_up - p_dn) / (2.0 * bump_T)
    theta = -dV_dT

    # Optional bump check for delta stability at the money (not returned)
    _ = bump_S
    return price, delta, theta
