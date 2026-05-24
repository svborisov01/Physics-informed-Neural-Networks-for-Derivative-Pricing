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
    if call_put == "Call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)
        
    disc_payoff = np.exp(-r * T) * payoff

    price = disc_payoff.mean()

    if return_stderr:
        stderr = disc_payoff.std(ddof=1) / np.sqrt(n_paths)
        return price, stderr

    return price