"""
2-factor Bergomi PINN option pricing in log-moneyness space.

Normalized price u = V / K with x = log(S / K). The network learns a
correction U on top of a Black–Scholes baseline with softplus positivity:

    u = softplus_β( u_BS(x, tau; r, sigma_eff) + U(x, X1, X2, tau, ...) )

Dynamics (risk-neutral, q = 0):

    dS/S = r dt + sqrt(v) dW^S
    dX1  = -κ1 X1 dt + dW1
    dX2  = -κ2 X2 dt + dW2

with correlations (ρ1, ρ2, ρ12) among (W^S, W1, W2).

Stationary instantaneous variance (flat initial forward variance ξ0):

    v(X1, X2) = ξ0 * exp(
        ω1 X1 + ω2 X2
        - ω1²/(4 κ1) - ω2²/(4 κ2)
        - ω1 ω2 ρ12 / (κ1 + κ2)
    )

PDE residual (stationary, q = 0):

    -u_τ + (r - ½v) u_x + ½v u_xx
    - κ1 X1 u_{X1} - κ2 X2 u_{X2}
    + ½ u_{X1 X1} + ½ u_{X2 X2} + ρ12 u_{X1 X2}
    + ρ1 √v u_{x X1} + ρ2 √v u_{x X2}
    - r u  = 0

Feature set (28): dual-factor OU state, both timescales, log-variance drive,
mean-reversion clocks, and key cross terms — see FEATURE_NAMES.
"""

from __future__ import annotations

import copy
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

from support_tools.analytical_pricing_tools import bs_option_normalized_from_x

torch.manual_seed(42)
np.random.seed(42)

DEFAULT_PRICE_SOFTPLUS_BETA = 400.0

FEATURE_NAMES = [
    "x_norm",
    "X1_norm",
    "X2_norm",
    "tau_norm",
    "r_norm",
    "xi0_norm",
    "omega1_norm",
    "omega2_norm",
    "kappa1_norm",
    "kappa2_norm",
    "rho1",
    "rho2",
    "rho12",
    "tau_sq",
    "tau_sin",
    "tau_cos",
    "log_v_drive_norm",
    "mr_fast",
    "mr_slow",
    "x_sq",
    "abs_x",
    "X1_sq",
    "X2_sq",
    "x_tau",
    "x_X1",
    "x_X2",
    "X1_X2",
    "cp_feat",
]


def _soft_positive_price(u, beta: float = DEFAULT_PRICE_SOFTPLUS_BETA):
    if beta is None or float(beta) <= 0.0:
        return u
    return torch.nn.functional.softplus(u, beta=float(beta))


def v_bergomi2(
    X1,
    X2,
    xi0,
    omega1,
    omega2,
    kappa1,
    kappa2,
    rho12,
    t=None,
    stationary: bool = True,
    eps: float = 1e-8,
    v_max: float | None = None,
):
    """
    Instantaneous variance under 2-factor Bergomi with flat initial curve.

    Stationary compensator uses
      Var(ω·X) = ω1²/(2κ1) + ω2²/(2κ2) + 2 ω1 ω2 ρ12/(κ1+κ2),
    so the exponent subtracts half of that.
    """
    if not isinstance(X1, torch.Tensor):
        X1 = torch.tensor(X1, dtype=torch.float32)

    def _like(z):
        if isinstance(z, torch.Tensor):
            return z.to(device=X1.device, dtype=X1.dtype).expand_as(X1)
        return torch.full_like(X1, float(z))

    X2_t = _like(X2)
    xi0_t = _like(xi0).clamp_min(eps)
    omega1_t = _like(omega1)
    omega2_t = _like(omega2)
    kappa1_t = _like(kappa1).clamp_min(max(eps, 1e-3))
    kappa2_t = _like(kappa2).clamp_min(max(eps, 1e-3))
    rho12_t = _like(rho12).clamp(-0.999, 0.999)

    drive = omega1_t * X1 + omega2_t * X2_t

    if stationary or t is None:
        compensator = (
            (omega1_t ** 2) / (4.0 * kappa1_t)
            + (omega2_t ** 2) / (4.0 * kappa2_t)
            + (omega1_t * omega2_t * rho12_t) / (kappa1_t + kappa2_t)
        )
    else:
        t_t = _like(t).clamp_min(0.0)
        # Exact finite-t compensator for correlated OU factors
        var1 = (1.0 - torch.exp(-2.0 * kappa1_t * t_t)) / (2.0 * kappa1_t)
        var2 = (1.0 - torch.exp(-2.0 * kappa2_t * t_t)) / (2.0 * kappa2_t)
        cov12 = (
            rho12_t
            * (1.0 - torch.exp(-(kappa1_t + kappa2_t) * t_t))
            / (kappa1_t + kappa2_t)
        )
        compensator = 0.5 * (
            (omega1_t ** 2) * var1
            + (omega2_t ** 2) * var2
            + 2.0 * omega1_t * omega2_t * cov12
        )

    expo = (drive - compensator).clamp(min=-20.0, max=20.0)
    v = (xi0_t * torch.exp(expo)).clamp_min(eps)
    if v_max is not None:
        v = v.clamp_max(float(v_max))
    return v


def sigma_bs_bergomi2(
    xi0,
    v=None,
    tau=None,
    mode: str = "spot_var",
    blend_c: float = 0.5,
    eps: float = 1e-8,
):
    """Effective BS vol for the 2-factor Bergomi baseline (same modes as 1F)."""
    if not isinstance(xi0, torch.Tensor):
        xi0 = torch.tensor(xi0, dtype=torch.float32)
    xi0 = xi0.clamp_min(eps)
    sigma_flat = torch.sqrt(xi0)
    mode = mode.lower()
    if mode == "flat_fwd":
        return sigma_flat
    if v is None:
        raise ValueError("v must be provided for mode='spot_var' or 'hybrid'")
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32, device=xi0.device)
    v = v.to(device=xi0.device, dtype=xi0.dtype).clamp_min(eps)
    sigma_spot = torch.sqrt(v)
    if mode == "spot_var":
        return sigma_spot
    if mode == "hybrid":
        if tau is None:
            raise ValueError("tau must be provided for mode='hybrid'")
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=xi0.dtype, device=xi0.device)
        tau = tau.to(device=xi0.device, dtype=xi0.dtype).clamp_min(eps)
        w = (tau / (tau + blend_c)).clamp(0.0, 1.0)
        return ((1.0 - w) * sigma_spot + w * sigma_flat).clamp_min(np.sqrt(eps))
    raise ValueError(f"Unknown sigma mode '{mode}'")


def _sample_correlations(N_points, device, dtype):
    """
    Sample (ρ1, ρ2, ρ12) so the 3×3 correlation matrix of (W^S, W1, W2) is PSD.
    """
    rho1 = -0.9 + 1.6 * torch.rand(N_points, 1, device=device, dtype=dtype)  # mostly neg
    rho2 = -0.7 + 1.4 * torch.rand(N_points, 1, device=device, dtype=dtype)
    # Bound |ρ12| by the PSD constraint roughly |ρ12| ≤ 1 and det ≥ 0.
    # Use a safe range and reject/clamp via Schur complement.
    rho12_raw = -0.6 + 1.2 * torch.rand(N_points, 1, device=device, dtype=dtype)
    # Require 1 - ρ1² - ρ2² - ρ12² + 2 ρ1 ρ2 ρ12 ≥ 0
    # Clamp ρ12 toward 0 until constraint holds (cheap projection).
    rho12 = rho12_raw.clone()
    for _ in range(4):
        det_like = (
            1.0
            - rho1 ** 2
            - rho2 ** 2
            - rho12 ** 2
            + 2.0 * rho1 * rho2 * rho12
        )
        bad = det_like < 1e-4
        rho12 = torch.where(bad, 0.5 * rho12, rho12)
    return rho1, rho2, rho12


def _sample_interior(pinn, N_points):
    """Sample interior collocation points for the 2-factor Bergomi PDE."""
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    kappa_floor = float(getattr(pinn, "kappa_floor", 0.25))
    kappa_floor = min(kappa_floor, 0.99 * float(pinn.kappa_max))
    kappa_floor = max(kappa_floor, 1e-3)

    # Prefer separated timescales: κ1 > κ2 on average (fast / slow)
    kappa1 = kappa_floor + (pinn.kappa_max - kappa_floor) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )
    kappa2 = kappa_floor + (pinn.kappa_max - kappa_floor) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )
    # Swap so kappa1 is the faster factor
    swap = kappa1 < kappa2
    kappa1, kappa2 = (
        torch.where(swap, kappa2, kappa1),
        torch.where(swap, kappa1, kappa2),
    )

    ou_std1 = 1.0 / torch.sqrt(2.0 * kappa1)
    ou_std2 = 1.0 / torch.sqrt(2.0 * kappa2)
    X1 = (ou_std1 * torch.randn(N_points, 1, device=device, dtype=dtype)).clamp(
        -pinn.X1_max, pinn.X1_max
    )
    X2 = (ou_std2 * torch.randn(N_points, 1, device=device, dtype=dtype)).clamp(
        -pinn.X2_max, pinn.X2_max
    )

    n_atm = N_points // 2
    n_unif = N_points - n_atm
    x_unif = pinn.x_min + (pinn.x_max - pinn.x_min) * torch.rand(
        n_unif, 1, device=device, dtype=dtype
    )
    x_atm = (0.15 * torch.randn(n_atm, 1, device=device, dtype=dtype)).clamp(
        pinn.x_min, pinn.x_max
    )
    x = torch.cat([x_unif, x_atm], dim=0)

    perm = torch.randperm(N_points, device=device)
    x = x[perm].detach().requires_grad_(True)
    X1 = X1[perm].detach().requires_grad_(True)
    X2 = X2[perm].detach().requires_grad_(True)
    kappa1 = kappa1[perm]
    kappa2 = kappa2[perm]

    tau = (
        1e-4 + (pinn.T - 1e-4) * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    r = 1e-4 + (pinn.r_max - 1e-4) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )
    xi0 = 1e-4 + (pinn.xi0_max - 1e-4) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )
    omega1 = 1e-4 + (pinn.omega_max - 1e-4) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )
    omega2 = 1e-4 + (pinn.omega_max - 1e-4) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )
    rho1, rho2, rho12 = _sample_correlations(N_points, device, dtype)

    return (
        x,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
    )


def _total_price(
    pinn,
    x,
    X1,
    X2,
    tau,
    r,
    xi0,
    omega1,
    omega2,
    kappa1,
    kappa2,
    rho1,
    rho2,
    rho12,
    sigma_mode,
    stationary,
):
    U = pinn.forward_x(
        x, X1, X2, tau, r, xi0, omega1, omega2, kappa1, kappa2, rho1, rho2, rho12
    )
    t = None if stationary else (pinn.T - tau).clamp_min(0.0)
    v = v_bergomi2(
        X1,
        X2,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho12,
        t=t,
        stationary=stationary,
        v_max=getattr(pinn, "v_max", 1.0),
    )
    sigma_bs = sigma_bs_bergomi2(xi0=xi0, v=v, tau=tau, mode=sigma_mode)
    u_bs = bs_option_normalized_from_x(
        x=x, tau=tau, r=r, sigma_bs=sigma_bs, call_put=pinn.call_put
    )
    beta = float(getattr(pinn, "price_softplus_beta", DEFAULT_PRICE_SOFTPLUS_BETA))
    u = _soft_positive_price(u_bs + U, beta=beta)
    return u, u_bs, v


def pde_dynamic_x(
    pinn,
    N_points=4000,
    sigma_mode="spot_var",
    stationary=True,
):
    """PDE residual loss on softplus(u_BS + U)."""
    (
        x,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
    ) = _sample_interior(pinn, N_points)

    u_total, u_bs, v = _total_price(
        pinn,
        x,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
        sigma_mode,
        stationary,
    )
    sqrt_v = torch.sqrt(v)

    def operator(f):
        f_tau = torch.autograd.grad(f.sum(), tau, create_graph=True)[0]
        f_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        f_X1 = torch.autograd.grad(f.sum(), X1, create_graph=True)[0]
        f_X2 = torch.autograd.grad(f.sum(), X2, create_graph=True)[0]

        f_xx = torch.autograd.grad(f_x.sum(), x, create_graph=True)[0]
        f_X1X1 = torch.autograd.grad(f_X1.sum(), X1, create_graph=True)[0]
        f_X2X2 = torch.autograd.grad(f_X2.sum(), X2, create_graph=True)[0]
        f_X1X2 = torch.autograd.grad(f_X1.sum(), X2, create_graph=True)[0]
        f_xX1 = torch.autograd.grad(f_x.sum(), X1, create_graph=True)[0]
        f_xX2 = torch.autograd.grad(f_x.sum(), X2, create_graph=True)[0]

        return (
            -f_tau
            + (r - 0.5 * v) * f_x
            + 0.5 * v * f_xx
            - kappa1 * X1 * f_X1
            - kappa2 * X2 * f_X2
            + 0.5 * f_X1X1
            + 0.5 * f_X2X2
            + rho12 * f_X1X2
            + rho1 * sqrt_v * f_xX1
            + rho2 * sqrt_v * f_xX2
            - r * f
        )

    residual = operator(u_total) / (1.0 + torch.abs(u_bs))
    return torch.mean(residual ** 2)


def spot_terminal_condition_x(
    pinn,
    N_points=1500,
    sigma_mode="spot_var",
    stationary=True,
):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype
    (
        x,
        X1,
        X2,
        _,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
    ) = _sample_interior(pinn, N_points)
    tau = torch.zeros(N_points, 1, device=device, dtype=dtype)
    tau_safe = torch.full_like(x, 1e-6)

    U = pinn.forward_x(
        x, X1, X2, tau, r, xi0, omega1, omega2, kappa1, kappa2, rho1, rho2, rho12
    )
    t = None if stationary else torch.full_like(x, float(pinn.T))
    v = v_bergomi2(
        X1,
        X2,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho12,
        t=t,
        stationary=stationary,
        v_max=getattr(pinn, "v_max", 1.0),
    )
    sigma_bs = sigma_bs_bergomi2(xi0=xi0, v=v, tau=tau_safe, mode=sigma_mode)
    u_bs = bs_option_normalized_from_x(
        x=x, tau=tau_safe, r=r, sigma_bs=sigma_bs, call_put=pinn.call_put
    )
    beta = float(getattr(pinn, "price_softplus_beta", DEFAULT_PRICE_SOFTPLUS_BETA))
    u_total = _soft_positive_price(u_bs + U, beta=beta)

    cp = pinn.call_put.lower()
    if cp == "call":
        payoff = torch.relu(torch.exp(x) - 1.0)
    elif cp == "put":
        payoff = torch.relu(1.0 - torch.exp(x))
    else:
        raise ValueError("pinn.call_put must be either 'Call' or 'Put'")
    return torch.mean(((u_total - payoff) / (1.0 + payoff)) ** 2)


def spot_boundary_conditions_x(
    pinn,
    N_points=1500,
    sigma_mode="spot_var",
    stationary=True,
    lambda_price_low=1.0,
    lambda_delta_low=0.0,
    lambda_price_high=1.0,
    lambda_delta_high=0.1,
):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype
    (
        _,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
    ) = _sample_interior(pinn, N_points)

    x_low = torch.full(
        (N_points, 1), pinn.x_min, device=device, dtype=dtype
    ).requires_grad_(True)
    x_high = torch.full(
        (N_points, 1), pinn.x_max, device=device, dtype=dtype
    ).requires_grad_(True)

    u_low, _, _ = _total_price(
        pinn,
        x_low,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
        sigma_mode,
        stationary,
    )
    u_high, _, _ = _total_price(
        pinn,
        x_high,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
        sigma_mode,
        stationary,
    )
    ux_low = torch.autograd.grad(u_low.sum(), x_low, create_graph=True)[0]
    ux_high = torch.autograd.grad(u_high.sum(), x_high, create_graph=True)[0]

    cp = pinn.call_put.lower()
    if cp == "call":
        target_price_low = torch.zeros_like(u_low)
        target_delta_low = torch.zeros_like(ux_low)
        target_price_high = torch.exp(x_high) - torch.exp(-r * tau)
        target_delta_high = torch.exp(x_high)
    elif cp == "put":
        target_price_low = torch.exp(-r * tau)
        target_delta_low = torch.zeros_like(ux_low)
        target_price_high = torch.zeros_like(u_high)
        target_delta_high = torch.zeros_like(ux_high)
    else:
        raise ValueError("pinn.call_put must be either 'Call' or 'Put'")

    loss_price_low = torch.mean(
        ((u_low - target_price_low) / (1.0 + torch.abs(target_price_low))) ** 2
    )
    loss_delta_low = torch.mean(
        ((ux_low - target_delta_low) / (1.0 + torch.abs(target_delta_low))) ** 2
    )
    loss_price_high = torch.mean(
        ((u_high - target_price_high) / (1.0 + torch.abs(target_price_high))) ** 2
    )
    loss_delta_high = torch.mean(
        ((ux_high - target_delta_high) / (1.0 + torch.abs(target_delta_high))) ** 2
    )
    return (
        lambda_price_low * loss_price_low
        + lambda_delta_low * loss_delta_low
        + lambda_price_high * loss_price_high
        + lambda_delta_high * loss_delta_high
    )


class PINN(nn.Module):
    """
    PINN for normalized 2-factor Bergomi correction:

        u ~= softplus_β( u_BS + U(x, X1, X2, τ, params) )

    Optimal 28-feature map documented in FEATURE_NAMES / bergomi2/README.md.
    Distinguishing attributes: X1_max, X2_max (vs X_max for 1-factor).
    """

    def __init__(
        self,
        x_min,
        x_max,
        X1_max,
        X2_max,
        T,
        r_max,
        xi0_max,
        omega_max,
        kappa_max,
        call_put="Call",
        hidden=96,
        depth=4,
        v_max=1.0,
        kappa_floor=0.25,
        price_softplus_beta=DEFAULT_PRICE_SOFTPLUS_BETA,
    ):
        super().__init__()
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.x_scale = max(abs(x_min), abs(x_max), 1e-6)
        self.X1_max = float(X1_max)
        self.X2_max = float(X2_max)
        self.T = float(T)
        self.r_max = float(r_max)
        self.xi0_max = float(xi0_max)
        self.omega_max = float(omega_max)
        self.kappa_max = float(kappa_max)
        self.v_max = float(v_max)
        self.kappa_floor = float(kappa_floor)
        self.price_softplus_beta = float(price_softplus_beta)
        self.call_put = call_put
        self.depth = depth
        self.hidden = hidden
        self.n_factors = 2

        input_dim = len(FEATURE_NAMES)
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def _cp_feature(self, x):
        cp = self.call_put.lower()
        if cp == "call":
            return torch.ones_like(x)
        if cp == "put":
            return -torch.ones_like(x)
        raise ValueError("call_put must be either 'Call' or 'Put'")

    def _features_from_x(
        self,
        x,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
    ):
        x_norm = x / self.x_scale
        X1_norm = X1 / self.X1_max
        X2_norm = X2 / self.X2_max
        tau_norm = tau / self.T
        r_norm = r / self.r_max
        xi0_norm = xi0 / self.xi0_max
        omega1_norm = omega1 / self.omega_max
        omega2_norm = omega2 / self.omega_max
        kappa1_norm = kappa1 / self.kappa_max
        kappa2_norm = kappa2 / self.kappa_max

        tau_sq = tau_norm ** 2
        tau_sin = torch.sin(torch.pi * tau_norm)
        tau_cos = torch.cos(torch.pi * tau_norm)

        k1 = kappa1.clamp_min(1e-6)
        k2 = kappa2.clamp_min(1e-6)
        log_v_drive = (
            omega1 * X1
            + omega2 * X2
            - (omega1 ** 2) / (4.0 * k1)
            - (omega2 ** 2) / (4.0 * k2)
            - (omega1 * omega2 * rho12) / (k1 + k2)
        )
        log_v_drive_norm = log_v_drive / (
            1.0 + self.omega_max * (self.X1_max + self.X2_max)
        )

        mr_fast = 1.0 - torch.exp(-torch.clamp(kappa1 * tau, 0.0, 50.0))
        mr_slow = 1.0 - torch.exp(-torch.clamp(kappa2 * tau, 0.0, 50.0))

        x_sq = x_norm ** 2
        abs_x = torch.abs(x_norm)
        X1_sq = X1_norm ** 2
        X2_sq = X2_norm ** 2
        x_tau = x_norm * tau_norm
        x_X1 = x_norm * X1_norm
        x_X2 = x_norm * X2_norm
        X1_X2 = X1_norm * X2_norm
        cp_feat = self._cp_feature(x)

        return torch.cat(
            [
                x_norm,
                X1_norm,
                X2_norm,
                tau_norm,
                r_norm,
                xi0_norm,
                omega1_norm,
                omega2_norm,
                kappa1_norm,
                kappa2_norm,
                rho1,
                rho2,
                rho12,
                tau_sq,
                tau_sin,
                tau_cos,
                log_v_drive_norm,
                mr_fast,
                mr_slow,
                x_sq,
                abs_x,
                X1_sq,
                X2_sq,
                x_tau,
                x_X1,
                x_X2,
                X1_X2,
                cp_feat,
            ],
            dim=1,
        )

    def forward_x(
        self,
        x,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
    ):
        features = self._features_from_x(
            x, X1, X2, tau, r, xi0, omega1, omega2, kappa1, kappa2, rho1, rho2, rho12
        )
        raw = self.net(features)
        # Gate by total vol-of-vol so U → 0 as (ω1, ω2) → 0 (BS limit)
        omega_gate = (
            (omega1.abs() + omega2.abs()) / max(2.0 * self.omega_max, 1e-6)
        ).clamp(0.0, 1.0)
        corr_scale = 0.15 * (0.5 + tau / self.T) * (1.0 + 0.25 * torch.abs(x))
        return omega_gate * corr_scale * torch.tanh(raw)

    def forward(
        self,
        S,
        K,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
    ):
        eps = 1e-8
        x = torch.log(torch.clamp(S, min=eps) / torch.clamp(K, min=eps))
        return self.forward_x(
            x, X1, X2, tau, r, xi0, omega1, omega2, kappa1, kappa2, rho1, rho2, rho12
        )

    @staticmethod
    def _to_col(z, device):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32)
        if z.dim() == 0:
            z = z.view(1, 1)
        elif z.dim() == 1:
            z = z.view(-1, 1)
        return z.to(device)

    def predict_normalized_price(
        self,
        S,
        K,
        X1,
        X2,
        tau,
        r,
        xi0,
        omega1,
        omega2,
        kappa1,
        kappa2,
        rho1,
        rho2,
        rho12,
        sigma_mode="spot_var",
        stationary=True,
    ):
        device = next(self.parameters()).device
        S_t = self._to_col(S, device)
        K_t = self._to_col(K, device)
        X1_t = self._to_col(X1, device)
        X2_t = self._to_col(X2, device)
        tau_t = self._to_col(tau, device)
        r_t = self._to_col(r, device)
        xi0_t = self._to_col(xi0, device)
        omega1_t = self._to_col(omega1, device)
        omega2_t = self._to_col(omega2, device)
        kappa1_t = self._to_col(kappa1, device)
        kappa2_t = self._to_col(kappa2, device)
        rho1_t = self._to_col(rho1, device)
        rho2_t = self._to_col(rho2, device)
        rho12_t = self._to_col(rho12, device)

        eps = 1e-8
        x_t = torch.log(torch.clamp(S_t, min=eps) / torch.clamp(K_t, min=eps))
        with torch.no_grad():
            U = self.forward_x(
                x_t,
                X1_t,
                X2_t,
                tau_t,
                r_t,
                xi0_t,
                omega1_t,
                omega2_t,
                kappa1_t,
                kappa2_t,
                rho1_t,
                rho2_t,
                rho12_t,
            )
            t = None if stationary else (self.T - tau_t).clamp_min(0.0)
            v = v_bergomi2(
                X1_t,
                X2_t,
                xi0_t,
                omega1_t,
                omega2_t,
                kappa1_t,
                kappa2_t,
                rho12_t,
                t=t,
                stationary=stationary,
                v_max=getattr(self, "v_max", None),
            )
            sigma_bs = sigma_bs_bergomi2(xi0=xi0_t, v=v, tau=tau_t, mode=sigma_mode)
            u_bs = bs_option_normalized_from_x(
                x=x_t, tau=tau_t, r=r_t, sigma_bs=sigma_bs, call_put=self.call_put
            )
            beta = float(
                getattr(self, "price_softplus_beta", DEFAULT_PRICE_SOFTPLUS_BETA)
            )
            return _soft_positive_price(u_bs + U, beta=beta)

    def predict_price(self, S, K, X1, X2, tau, r, xi0, omega1, omega2, kappa1, kappa2, rho1, rho2, rho12, sigma_mode="spot_var", stationary=True):
        device = next(self.parameters()).device
        K_t = self._to_col(K, device)
        u_hat = self.predict_normalized_price(
            S=S,
            K=K,
            X1=X1,
            X2=X2,
            tau=tau,
            r=r,
            xi0=xi0,
            omega1=omega1,
            omega2=omega2,
            kappa1=kappa1,
            kappa2=kappa2,
            rho1=rho1,
            rho2=rho2,
            rho12=rho12,
            sigma_mode=sigma_mode,
            stationary=stationary,
        )
        return K_t * u_hat


def train_network(
    pinn,
    N_pde=3500,
    N_boundary=1200,
    epochs=2500,
    lr=1e-4,
    sigma_mode="spot_var",
    stationary=True,
    lambda_boundary=1.0,
    lambda_terminal=1.0,
    grad_clip=1.0,
    print_every=100,
    best_model_path="trained_models/bergomi2.pt",
    save_model=True,
    weight_decay=1e-2,
    cosine_eta_min=1e-5,
):
    device = next(pinn.parameters()).device
    optimizer = torch.optim.AdamW(
        pinn.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1500, eta_min=cosine_eta_min
    )

    history = {
        "epoch": [],
        "elapsed_time": [],
        "total": [],
        "physics": [],
        "terminal": [],
        "boundary": [],
        "lr": [],
        "model depth": pinn.depth,
        "model width": pinn.hidden,
        "sigma_mode": sigma_mode,
        "stationary": stationary,
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
    }

    best_loss = float("inf")
    best_epoch = None
    best_state_dict = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        pinn.train()
        optimizer.zero_grad(set_to_none=True)

        loss_physics = pde_dynamic_x(
            pinn, N_points=N_pde, sigma_mode=sigma_mode, stationary=stationary
        )
        loss_terminal = spot_terminal_condition_x(
            pinn, N_points=N_boundary, sigma_mode=sigma_mode, stationary=stationary
        )
        loss_boundary = spot_boundary_conditions_x(
            pinn, N_points=N_boundary, sigma_mode=sigma_mode, stationary=stationary
        )
        loss = (
            loss_physics
            + lambda_terminal * loss_terminal
            + lambda_boundary * loss_boundary
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")

        loss.backward()
        grad_norm = None
        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(pinn.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        current_loss = loss.item()

        history["epoch"].append(epoch)
        history["elapsed_time"].append(elapsed)
        history["total"].append(current_loss)
        history["physics"].append(loss_physics.item())
        history["terminal"].append(loss_terminal.item())
        history["boundary"].append(loss_boundary.item())
        history["lr"].append(current_lr)

        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(pinn.state_dict())
            if save_model:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss,
                        "sigma_mode": sigma_mode,
                        "stationary": stationary,
                        "call_put": pinn.call_put,
                        "x_min": pinn.x_min,
                        "x_max": pinn.x_max,
                        "X1_max": pinn.X1_max,
                        "X2_max": pinn.X2_max,
                        "T": pinn.T,
                        "r_max": pinn.r_max,
                        "xi0_max": pinn.xi0_max,
                        "omega_max": pinn.omega_max,
                        "kappa_max": pinn.kappa_max,
                        "v_max": pinn.v_max,
                        "kappa_floor": pinn.kappa_floor,
                        "price_softplus_beta": pinn.price_softplus_beta,
                        "positivity": "softplus",
                        "n_factors": 2,
                        "model_type": "bergomi2",
                        "hidden": pinn.hidden,
                        "depth": pinn.depth,
                        "feature_names": FEATURE_NAMES,
                    },
                    best_model_path,
                )

        if epoch % print_every == 0 or epoch == 1:
            gn = f"{float(grad_norm):.3e}" if grad_norm is not None else "None"
            print(
                f"Epoch {epoch:6d} | Time: {elapsed:10.2f}s | LR: {current_lr:.3e} | "
                f"Total: {current_loss:.6e} | PDE: {loss_physics.item():.6e} | "
                f"Term: {loss_terminal.item():.6e} | Bnd: {loss_boundary.item():.6e} | "
                f"Grad: {gn} | Best: {best_loss:.6e} @ {best_epoch}"
            )

    history["best_loss"] = best_loss
    history["best_epoch"] = best_epoch
    if best_state_dict is not None:
        pinn.load_state_dict(best_state_dict)
    return history
