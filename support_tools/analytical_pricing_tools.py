import torch
import numpy as np


def bs_option_torch(S, K, tau, r, sigma_bs, call_put="Call"):
    """
    Black-Scholes European option price in PyTorch.
    Supports call and put.
    tau is time-to-maturity, scalar or tensor, same shape as S.
    """
    eps = 1e-8
    S_clamp = torch.clamp(S, min=eps)
    K_clamp = torch.clamp(K, min=eps)
    tau_clamp = torch.clamp(tau, min=eps)
    sigma_clamp = torch.clamp(sigma_bs, min=eps)

    sig_sqrt_t = sigma_clamp * torch.sqrt(tau_clamp)
    d1 = (torch.log(S_clamp / K_clamp) + (r + 0.5 * sigma_clamp**2) * tau_clamp) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t

    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=S_clamp.device, dtype=S_clamp.dtype))
    Phi = lambda x: 0.5 * (1.0 + torch.erf(x / sqrt_2))

    cp = call_put.lower()
    if cp == "call":
        price = S_clamp * Phi(d1) - K_clamp * torch.exp(-r * tau_clamp) * Phi(d2)
    elif cp == "put":
        price = K_clamp * torch.exp(-r * tau_clamp) * Phi(-d2) - S_clamp * Phi(-d1)
    else:
        raise ValueError("call_put must be either 'Call' or 'Put'")

    return price


def bs_option_normalized_from_x(x, tau, r, sigma_bs, call_put="Call"):
    """
    Normalized BS price u = V / K as a function of x = log(S/K).
    Since S = K * exp(x), setting K = 1 gives the normalized price directly.
    """
    S_over_K = torch.exp(x)
    S = S_over_K
    K = torch.ones_like(S)
    V_bs = bs_option_torch(S, K, tau, r, sigma_bs, call_put=call_put)
    return V_bs


def sigma_bs_effective(
    v,
    theta=None,
    kappa=None,
    tau=None,
    mode: str = "hybrid",
    eps: float = 1e-6,
    blend_c: float = 0.5,
    blend_a: float = 10.0
):
    """
    Numerically stable effective Black-Scholes volatility for Heston-baseline PINN.

    Parameters
    ----------
    v : torch.Tensor
        Instantaneous variance state.
    theta : torch.Tensor or scalar, optional
        Long-run variance.
    kappa : torch.Tensor or scalar, optional
        Mean reversion speed.
    tau : torch.Tensor or scalar, optional
        Time-to-maturity.
    mode : {"spot_var", "mean_reverting", "hybrid"}
    eps : float
        Numerical floor.
    blend_c : float
        Time blending parameter for hybrid mode.
    blend_a : float
        Strength of dependence on |v-theta| for hybrid mode.

    Returns
    -------
    sigma_bs : torch.Tensor
        Effective Black-Scholes volatility.
    """

    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    device = v.device
    dtype = v.dtype

    v = v.clamp_min(eps)

    def to_tensor_like(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype).expand_as(v)
        return torch.full_like(v, float(x))

    if mode == "spot_var":
        v_eff = v

    elif mode == "mean_reverting":
        if theta is None or kappa is None or tau is None:
            raise ValueError("theta, kappa, and tau must be provided for mode='mean_reverting'")

        theta_t = to_tensor_like(theta).clamp_min(eps)
        kappa_t = to_tensor_like(kappa).clamp_min(eps)
        tau_t = to_tensor_like(tau).clamp_min(eps)

        x = (kappa_t * tau_t).clamp_min(eps)

        # Stable computation of (1 - exp(-x)) / x
        frac = -torch.expm1(-x) / x

        v_eff = theta_t + (v - theta_t) * frac
        v_eff = v_eff.clamp_min(eps)

    elif mode == "hybrid":
        if theta is None or kappa is None or tau is None:
            raise ValueError("theta, kappa, and tau must be provided for mode='hybrid'")

        theta_t = to_tensor_like(theta).clamp_min(eps)
        kappa_t = to_tensor_like(kappa).clamp_min(eps)
        tau_t = to_tensor_like(tau).clamp_min(eps)

        x = (kappa_t * tau_t).clamp_min(eps)

        # Stable mean-reverting fraction
        frac = -torch.expm1(-x) / x

        v_mr = theta_t + (v - theta_t) * frac
        v_mr = v_mr.clamp_min(eps)

        # Hybrid weight: more mean-reverting for longer tau and larger |v-theta|
        w_tau = tau_t / (tau_t + blend_c)
        w_gap = 1.0 - torch.exp(-blend_a * torch.abs(v - theta_t))
        w = (w_tau * w_gap).clamp(0.0, 1.0)

        v_eff = (1.0 - w) * v + w * v_mr
        v_eff = v_eff.clamp_min(eps)

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    sigma_bs = torch.sqrt(v_eff)
    sigma_bs = sigma_bs.clamp_min(np.sqrt(eps))

    return sigma_bs


# ============================================================
# Heston pricing via the Fang–Oosterlee COS method
# ============================================================


def heston_char_func(u, T, r, q, kappa, theta, sigma, rho, v0, S0):
    """
    Heston characteristic function of X = log(S_T).

    Uses the Albrecher et al. "Little Heston Trap" formulation for
    numerical stability. Fully vectorized over complex frequencies ``u``.
    """
    u = np.asarray(u, dtype=np.complex128)
    i = 1j

    a = kappa * theta
    b = kappa - rho * sigma * i * u
    d = np.sqrt(b * b + sigma * sigma * (i * u + u * u))

    # Little Heston Trap: g = (b - d) / (b + d)
    g = (b - d) / (b + d)
    exp_dt = np.exp(-d * T)

    G = (1.0 - g * exp_dt) / (1.0 - g)
    C = (r - q) * i * u * T + (a / (sigma * sigma)) * (
        (b - d) * T - 2.0 * np.log(G)
    )
    D = ((b - d) / (sigma * sigma)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

    return np.exp(C + D * v0 + i * u * np.log(S0))


def _heston_cumulants(T, r, q, kappa, theta, sigma, rho, v0):
    """
    First two cumulants of X = log(S_T / S_0) under Heston.

    Used to set the COS truncation interval [a, b].
    """
    c1 = (
        (r - q) * T
        + (1.0 - np.exp(-kappa * T)) * (theta - v0) / (2.0 * kappa)
        - 0.5 * theta * T
    )

    exp_kt = np.exp(-kappa * T)
    exp_2kt = np.exp(-2.0 * kappa * T)
    sigma2 = sigma * sigma
    kappa2 = kappa * kappa
    kappa3 = kappa2 * kappa

    c2 = (1.0 / (8.0 * kappa3)) * (
        sigma * T * kappa * exp_kt * (v0 - theta) * (8.0 * kappa * rho - 4.0 * sigma)
        + kappa * rho * sigma * (1.0 - exp_kt) * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * T * (-4.0 * kappa * rho * sigma + sigma2 + 4.0 * kappa2)
        + sigma2
        * (
            (theta - 2.0 * v0) * exp_2kt
            + theta * (6.0 * exp_kt - 7.0)
            + 2.0 * v0
        )
        + 8.0 * kappa2 * (v0 - theta) * (1.0 - exp_kt)
    )

    return c1, max(float(c2), 1e-12)


def _cos_chi_psi(a, b, c, d, k):
    """
    Analytic integrals χ_k(c,d) and ψ_k(c,d) for European payoffs
    on the COS truncation interval [a, b].
    """
    k = np.asarray(k, dtype=np.float64)
    omega = k * np.pi / (b - a)

    chi = np.empty_like(omega, dtype=np.float64)
    psi = np.empty_like(omega, dtype=np.float64)
    mask0 = np.abs(omega) < 1e-14
    mask = ~mask0

    chi[mask0] = np.exp(d) - np.exp(c)
    chi[mask] = (
        1.0
        / (1.0 + omega[mask] ** 2)
        * (
            np.cos(omega[mask] * (d - a)) * np.exp(d)
            - np.cos(omega[mask] * (c - a)) * np.exp(c)
            + omega[mask]
            * (
                np.sin(omega[mask] * (d - a)) * np.exp(d)
                - np.sin(omega[mask] * (c - a)) * np.exp(c)
            )
        )
    )

    psi[mask0] = d - c
    psi[mask] = (
        1.0
        / omega[mask]
        * (np.sin(omega[mask] * (d - a)) - np.sin(omega[mask] * (c - a)))
    )

    return chi, psi


def _cos_payoff_coefficients(a, b, K, call_put, N):
    """Fourier-cosine coefficients V_k of the European payoff."""
    k = np.arange(N, dtype=np.float64)
    cp = call_put.lower()

    if cp == "call":
        # payoff K max(e^x - 1, 0) with x = log(S/K); support [0, b]
        c, d = 0.0, b
        chi, psi = _cos_chi_psi(a, b, c, d, k)
        Uk = (2.0 / (b - a)) * K * (chi - psi)
    elif cp == "put":
        # support [a, 0]
        c, d = a, 0.0
        chi, psi = _cos_chi_psi(a, b, c, d, k)
        Uk = (2.0 / (b - a)) * K * (-chi + psi)
    else:
        raise ValueError("call_put must be either 'Call' or 'Put'")

    return Uk


def heston_price(
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
    N=256,
    L=10.0,
    verbose=False,
    **_ignored,
):
    """
    Heston European option price via the Fang–Oosterlee COS method.

    Parameters
    ----------
    S0, K, T, r, q : float
        Spot, strike, maturity, risk-free rate, dividend yield.
    v0, kappa, theta, sigma, rho : float
        Heston parameters (initial variance, mean reversion, long-run
        variance, vol-of-vol, correlation).
    call_put : {"Call", "Put"}
    N : int
        Number of cosine terms (256 is typically enough for equity options).
    L : float
        Truncation-range multiplier in units of sqrt(c2).
    verbose : bool
        Print price and truncation diagnostics when True.
    **_ignored
        Accepted for backward compatibility with the old Gil-Pelaez kwargs
        (``integration_limit``, ``epsabs``, ``epsrel``, ``limit``).

    Returns
    -------
    price : float
    truncation_halfwidth : float
        Half-width of the COS truncation interval (diagnostic).
    diagnostics : dict
    """
    cp = call_put.lower()
    if cp not in {"call", "put"}:
        raise ValueError("call_put must be either 'Call' or 'Put'")

    S0 = float(S0)
    K = float(K)
    T = float(T)
    r = float(r)
    q = float(q)
    v0 = float(v0)
    kappa = float(kappa)
    theta = float(theta)
    sigma = float(sigma)
    rho = float(rho)
    N = int(N)
    L = float(L)

    if T <= 0.0:
        intrinsic = max(S0 - K, 0.0) if cp == "call" else max(K - S0, 0.0)
        diagnostics = {"method": "intrinsic", "N": N, "a": np.nan, "b": np.nan}
        return float(intrinsic), 0.0, diagnostics

    # Cumulants of log(S_T / S_0); shift by log(S0/K) for the x = log(S/K) grid
    c1, c2 = _heston_cumulants(T, r, q, kappa, theta, sigma, rho, v0)
    x = np.log(S0 / K)
    a = (x + c1) - L * np.sqrt(c2)
    b = (x + c1) + L * np.sqrt(c2)

    k = np.arange(N, dtype=np.float64)
    u = k * np.pi / (b - a)

    # CF of log(S_T), then CF of x_T = log(S_T / K)
    phi = heston_char_func(u, T, r, q, kappa, theta, sigma, rho, v0, S0)
    phi_x = phi * np.exp(-1j * u * np.log(K))

    Uk = _cos_payoff_coefficients(a, b, K, call_put=cp, N=N)

    # First term of the cosine series is halved
    weights = np.ones(N)
    weights[0] = 0.5

    terms = weights * np.real(phi_x * np.exp(-1j * u * a) * Uk)
    price = float(np.exp(-r * T) * np.sum(terms))
    price = max(price, 0.0)

    truncation_halfwidth = L * np.sqrt(c2)
    diagnostics = {
        "method": "COS",
        "N": N,
        "L": L,
        "a": a,
        "b": b,
        "c1": c1,
        "c2": c2,
        "x": x,
        "truncation_halfwidth": truncation_halfwidth,
    }

    if verbose:
        print(f"Heston {call_put} price (COS): {price:.8f}")
        print(f"truncation interval: [{a:.4f}, {b:.4f}]  (N={N}, L={L})")

    return price, truncation_halfwidth, diagnostics


def heston_call_price(*args, **kwargs):
    kwargs["call_put"] = "Call"
    price, trunc, diagnostics = heston_price(*args, **kwargs)
    return price, trunc, diagnostics


def heston_put_price(*args, **kwargs):
    kwargs["call_put"] = "Put"
    price, trunc, diagnostics = heston_price(*args, **kwargs)
    return price, trunc, diagnostics
