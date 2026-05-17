import torch
import numpy as np
from scipy.integrate import quad

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
        tau_t   = to_tensor_like(tau).clamp_min(eps)

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
        tau_t   = to_tensor_like(tau).clamp_min(eps)

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

def heston_char_func(u, tau, S0, r, q, v0, kappa, theta, sigma, rho, j=2):
    """
    Heston characteristic function - Little Heston Trap formulation.
    j=1 for P1, j=2 for P2.
    """
    i = 1j

    u1 = 0.5
    u2 = -0.5
    uj = u1 if j == 1 else u2

    a = kappa * theta
    b = kappa

    delta = b - rho * sigma * u * i
    discriminant = delta**2 - sigma**2 * (2.0 * uj * u * i - u**2)

    d = np.sqrt(discriminant + 0j)

    if np.abs(d) < 1e-10:
        d = 1e-10 + 0j

    denom = delta + d
    if np.abs(denom) < 1e-12:
        return 0.0 + 0j

    g = (delta - d) / denom

    exp_term = np.exp(-d * tau)
    one_minus_g_exp = 1.0 - g * exp_term
    one_minus_g = 1.0 - g

    if np.abs(one_minus_g_exp) < 1e-12 or np.abs(one_minus_g) < 1e-12:
        return 0.0 + 0j

    D = ((delta - d) / sigma**2) * ((1.0 - exp_term) / one_minus_g_exp)

    log_arg = one_minus_g_exp / one_minus_g
    if np.abs(log_arg) < 1e-12:
        return 0.0 + 0j

    log_val = np.log(log_arg + 0j)

    C = (r - q) * u * i * tau + (a / sigma**2) * ((delta - d) * tau - 2.0 * log_val)

    cf = np.exp(C + D * v0 + i * u * np.log(S0))

    if np.isnan(cf.real) or np.isnan(cf.imag) or np.isinf(cf.real) or np.isinf(cf.imag):
        return 0.0 + 0j

    return cf


def heston_prob(j, S0, K, T, r, q, v0, kappa, theta, sigma, rho,
                integration_limit=100.0, epsabs=1e-6, epsrel=1e-6, limit=200):
    """
    Compute Heston probability P1 or P2 by Gil-Pelaez inversion.
    Returns:
        Pj, integral_value, integral_abs_error
    """
    tau = T
    logK = np.log(K)

    def integrand(u):
        if u < 1e-10:
            return 0.0

        cf = heston_char_func(u, tau, S0, r, q, v0, kappa, theta, sigma, rho, j=j)
        val = np.exp(-1j * u * logK) * cf / (1j * u)
        out = np.real(val)

        if np.isnan(out) or np.isinf(out):
            return 0.0
        return out

    integral, abs_err = quad(
        integrand,
        1e-8,
        integration_limit,
        limit=limit,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    Pj = 0.5 + integral / np.pi
    Pj = np.clip(Pj, 0.0, 1.0)

    return Pj, integral, abs_err


def heston_price(
    S0, K, T, r, q, v0, kappa, theta, sigma, rho,
    call_put="Call",
    integration_limit=100.0,
    epsabs=1e-6,
    epsrel=1e-6,
    limit=200,
    verbose=False,
):
    """
    Heston European option price using Gil-Pelaez inversion.

    Supports both calls and puts.
    Also reports numerical integration absolute error estimates from quad().

    Returns
    -------
    price : float
    price_std_err : float
        Error proxy from quad absolute error estimates, propagated to price.
        This is not a statistical Monte Carlo standard error, but a useful
        numerical integration uncertainty estimate.
    diagnostics : dict
    """
    cp = call_put.lower()
    if cp not in {"call", "put"}:
        raise ValueError("call_put must be either 'Call' or 'Put'")

    try:
        P1, P1_int, P1_err = heston_prob(
            1, S0, K, T, r, q, v0, kappa, theta, sigma, rho,
            integration_limit=integration_limit, epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        P2, P2_int, P2_err = heston_prob(
            2, S0, K, T, r, q, v0, kappa, theta, sigma, rho,
            integration_limit=integration_limit, epsabs=epsabs, epsrel=epsrel, limit=limit
        )
    except Exception as e:
        if verbose:
            print(f"Integration failed - returning NaN ({e})")
        diagnostics = {
            "P1": np.nan, "P2": np.nan,
            "P1_integral": np.nan, "P2_integral": np.nan,
            "P1_abs_error": np.nan, "P2_abs_error": np.nan,
            "price_abs_error_proxy": np.nan,
        }
        return np.nan, np.nan, diagnostics

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    call_price = disc_q * S0 * P1 - disc_r * K * P2

    # Put from put-call parity for European options
    # C - P = S0 e^{-qT} - K e^{-rT}
    if cp == "call":
        price = call_price
    else:
        price = call_price - disc_q * S0 + disc_r * K

    # Propagate quad absolute error estimates into a price error proxy.
    # Since Pj = 0.5 + integral/pi, dPj error ~= err/pi.
    P1_prob_err = P1_err / np.pi
    P2_prob_err = P2_err / np.pi

    # Conservative quadrature-combined price uncertainty
    call_price_abs_err = np.sqrt(
        (disc_q * S0 * P1_prob_err) ** 2 +
        (disc_r * K * P2_prob_err) ** 2
    )

    # Put via parity uses same quadrature uncertainty as call,
    # since parity adjustment is deterministic.
    price_abs_err = call_price_abs_err

    diagnostics = {
        "P1": P1,
        "P2": P2,
        "P1_integral": P1_int,
        "P2_integral": P2_int,
        "P1_abs_error": P1_err,
        "P2_abs_error": P2_err,
        "P1_prob_abs_error": P1_prob_err,
        "P2_prob_abs_error": P2_prob_err,
        "call_price": call_price,
        "price_abs_error_proxy": price_abs_err,
    }

    if verbose:
        print(f"Heston {call_put} price: {price:.8f}")
        print(f"quad abs error P1 integral: {P1_err:.3e}")
        print(f"quad abs error P2 integral: {P2_err:.3e}")
        print(f"price abs error proxy:      {price_abs_err:.3e}")

    return price, price_abs_err, diagnostics


def heston_call_price(*args, **kwargs):
    kwargs["call_put"] = "Call"
    price, price_err, diagnostics = heston_price(*args, **kwargs)
    return price, price_err, diagnostics


def heston_put_price(*args, **kwargs):
    kwargs["call_put"] = "Put"
    price, price_err, diagnostics = heston_price(*args, **kwargs)
    return price, price_err, diagnostics