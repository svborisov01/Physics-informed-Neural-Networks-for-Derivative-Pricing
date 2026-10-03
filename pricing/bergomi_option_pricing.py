"""
1-factor Bergomi PINN option pricing in log-moneyness space.

Normalized price u = V / K with x = log(S / K). The network learns a
correction U on top of a Black–Scholes baseline:

    u = u_BS(x, tau; r, sigma_eff) + U(x, X, tau, ...)

Instantaneous variance (flat initial forward variance xi0):

    v(t, X) = xi0 * exp( omega * X - (omega^2 / (4*kappa)) * (1 - exp(-2*kappa*t)) )

or the stationary approximation (default):

    v(X) = xi0 * exp( omega * X - omega^2 / (4*kappa) )

PDE residual (q = 0):

    -u_tau + (r - 0.5*v) u_x + 0.5*v u_xx
    - kappa*X u_X + 0.5 u_XX + rho*sqrt(v) u_xX - r*u  = 0

See docs/bergomi_1factor_pde.md for the full derivation.
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


# ============================================================
# Variance and BS baseline helpers
# ============================================================


def v_bergomi(
    X,
    xi0,
    omega,
    kappa,
    t=None,
    stationary: bool = True,
    eps: float = 1e-8,
    v_max: float | None = None,
):
    """
    Instantaneous variance under 1-factor Bergomi with flat initial curve.

    Parameters
    ----------
    X : tensor
        OU forward-variance factor.
    xi0, omega, kappa : tensor or float
        Flat forward variance, vol-of-variance, mean-reversion speed.
    t : tensor or None
        Calendar time. Ignored when ``stationary=True``.
    stationary : bool
        If True, use Var_infty(X) = 1/(2 kappa).
    v_max : float or None
        Optional upper clamp on variance (stabilizes training).
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)

    def _like(z):
        if isinstance(z, torch.Tensor):
            return z.to(device=X.device, dtype=X.dtype).expand_as(X)
        return torch.full_like(X, float(z))

    xi0_t = _like(xi0).clamp_min(eps)
    omega_t = _like(omega)
    # Floor kappa so the stationary compensator omega^2/(4 kappa) stays finite
    kappa_t = _like(kappa).clamp_min(max(eps, 1e-3))

    if stationary or t is None:
        expo = omega_t * X - (omega_t ** 2) / (4.0 * kappa_t)
    else:
        t_t = _like(t).clamp_min(0.0)
        expo = omega_t * X - (omega_t ** 2) / (4.0 * kappa_t) * (
            1.0 - torch.exp(-2.0 * kappa_t * t_t)
        )

    # Cap the exponent to avoid overflow before clamping v
    expo = expo.clamp(min=-20.0, max=20.0)
    v = (xi0_t * torch.exp(expo)).clamp_min(eps)
    if v_max is not None:
        v = v.clamp_max(float(v_max))
    return v


def sigma_bs_bergomi(
    xi0,
    v=None,
    tau=None,
    mode: str = "flat_fwd",
    blend_c: float = 0.5,
    eps: float = 1e-8,
):
    """
    Effective Black–Scholes volatility for the Bergomi baseline.

    Modes
    -----
    flat_fwd : sigma = sqrt(xi0)   (exact Black–Scholes limit as omega -> 0)
    spot_var : sigma = sqrt(v)     (default for PINN training; better ATM baseline)
    hybrid   : blend flat_fwd and spot_var with tau-dependent weight
    """
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
        # near expiry prefer spot var; longer tau prefer flat forward
        sigma = (1.0 - w) * sigma_spot + w * sigma_flat
        return sigma.clamp_min(np.sqrt(eps))

    raise ValueError(f"Unknown sigma mode '{mode}'. Use flat_fwd, spot_var, or hybrid.")


# ============================================================
# Collocation sampling
# ============================================================


def _sample_interior(pinn, N_points):
    """
    Sample (x, X, tau, r, xi0, omega, kappa, rho) in the training domain.

    Only (x, X, tau) require gradients for the PDE operator. The OU factor X is
    drawn from a truncated stationary Gaussian N(0, 1/(2 kappa)) so that the
    implied variance v(X) stays in a realistic range (unlike uniform X x omega
    sampling, which produced v up to O(10-100) and broke training).
    """
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    # Mean-reversion with a meaningful floor (avoids omega^2/(4 kappa) blow-ups)
    kappa_floor = float(getattr(pinn, "kappa_floor", 0.25))
    kappa_floor = min(kappa_floor, 0.99 * float(pinn.kappa_max))
    kappa_floor = max(kappa_floor, 1e-3)
    kappa = kappa_floor + (pinn.kappa_max - kappa_floor) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )

    # Stationary OU scale: std = 1/sqrt(2 kappa)
    ou_std = 1.0 / torch.sqrt(2.0 * kappa)
    X = (ou_std * torch.randn(N_points, 1, device=device, dtype=dtype)).clamp(
        -pinn.X_max, pinn.X_max
    )

    # Oversample near ATM (x≈0): absolute pricing error and payoff kink concentrate there
    n_atm = N_points // 2
    n_unif = N_points - n_atm
    x_unif = pinn.x_min + (pinn.x_max - pinn.x_min) * torch.rand(
        n_unif, 1, device=device, dtype=dtype
    )
    x_atm = (0.15 * torch.randn(n_atm, 1, device=device, dtype=dtype)).clamp(
        pinn.x_min, pinn.x_max
    )
    x = torch.cat([x_unif, x_atm], dim=0)

    # Shuffle so PDE / BC batches are mixed (keep X paired with its kappa)
    perm = torch.randperm(N_points, device=device)
    x = x[perm].detach().requires_grad_(True)
    X = X[perm].detach().requires_grad_(True)
    kappa = kappa[perm]

    tau = (
        1e-4 + (pinn.T - 1e-4) * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    r = 1e-4 + (pinn.r_max - 1e-4) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )
    xi0 = 1e-4 + (pinn.xi0_max - 1e-4) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )
    omega = 1e-4 + (pinn.omega_max - 1e-4) * torch.rand(
        N_points, 1, device=device, dtype=dtype
    )
    rho = -0.95 + 1.90 * torch.rand(N_points, 1, device=device, dtype=dtype)

    return x, X, tau, r, xi0, omega, kappa, rho


def _soft_positive_price(u, beta: float = 40.0):
    """
    Soft positivity for normalized European prices.

    Uses softplus (not softmax): softplus_β(u) = log(1+exp(βu))/β > 0,
    ≈ u for moderate/large positive u. beta <= 0 disables the map.
    """
    if beta is None or float(beta) <= 0.0:
        return u
    return torch.nn.functional.softplus(u, beta=float(beta))


def _total_price(pinn, x, X, tau, r, xi0, omega, kappa, rho, sigma_mode, stationary):
    """Build u = softplus(u_BS + U) at collocation points (if enabled)."""
    U = pinn.forward_x(x, X, tau, r, xi0, omega, kappa, rho)

    if stationary:
        t = None
    else:
        # Convention: option issued at calendar t=0 with maturity pinn.T
        t = (pinn.T - tau).clamp_min(0.0)

    v = v_bergomi(
        X,
        xi0,
        omega,
        kappa,
        t=t,
        stationary=stationary,
        v_max=getattr(pinn, "v_max", 1.0),
    )
    sigma_bs = sigma_bs_bergomi(xi0=xi0, v=v, tau=tau, mode=sigma_mode)
    u_bs = bs_option_normalized_from_x(
        x=x, tau=tau, r=r, sigma_bs=sigma_bs, call_put=pinn.call_put
    )
    u_raw = u_bs + U
    beta = float(getattr(pinn, "price_softplus_beta", 0.0))
    u = _soft_positive_price(u_raw, beta=beta)
    return u, u_bs, v


# ============================================================
# Loss terms
# ============================================================


def pde_dynamic_x(
    pinn,
    N_points=5000,
    sigma_mode="spot_var",
    stationary=True,
    detach_source=False,
):
    """PDE residual loss on u = u_BS + U."""
    x, X, tau, r, xi0, omega, kappa, rho = _sample_interior(pinn, N_points)

    u_total, u_bs, v = _total_price(
        pinn, x, X, tau, r, xi0, omega, kappa, rho, sigma_mode, stationary
    )

    if detach_source:
        u_bs = u_bs.detach()
        # rebuild total with detached baseline so residual targets U mainly
        U = pinn.forward_x(x, X, tau, r, xi0, omega, kappa, rho)
        u_total = u_bs + U

    def bergomi_operator(f):
        f_tau = torch.autograd.grad(f.sum(), tau, create_graph=True)[0]
        f_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        f_X = torch.autograd.grad(f.sum(), X, create_graph=True)[0]

        f_xx = torch.autograd.grad(f_x.sum(), x, create_graph=True)[0]
        f_XX = torch.autograd.grad(f_X.sum(), X, create_graph=True)[0]
        f_xX = torch.autograd.grad(f_x.sum(), X, create_graph=True)[0]

        sqrt_v = torch.sqrt(v)
        return (
            -f_tau
            + (r - 0.5 * v) * f_x
            + 0.5 * v * f_xx
            - kappa * X * f_X
            + 0.5 * f_XX
            + rho * sqrt_v * f_xX
            - r * f
        )

    residual = bergomi_operator(u_total)
    residual = residual / (1.0 + torch.abs(u_bs))
    return torch.mean(residual ** 2)


def spot_terminal_condition_x(
    pinn,
    N_points=2000,
    sigma_mode="spot_var",
    stationary=True,
):
    """Terminal payoff loss at tau = 0."""
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    x, X, _, r, xi0, omega, kappa, rho = _sample_interior(pinn, N_points)
    tau = torch.zeros(N_points, 1, device=device, dtype=dtype)
    tau_safe = torch.full_like(x, 1e-6)

    U = pinn.forward_x(x, X, tau, r, xi0, omega, kappa, rho)

    if stationary:
        t = None
    else:
        t = torch.full_like(x, float(pinn.T))

    v = v_bergomi(
        X,
        xi0,
        omega,
        kappa,
        t=t,
        stationary=stationary,
        v_max=getattr(pinn, "v_max", 1.0),
    )
    sigma_bs = sigma_bs_bergomi(xi0=xi0, v=v, tau=tau_safe, mode=sigma_mode)
    u_bs = bs_option_normalized_from_x(
        x=x, tau=tau_safe, r=r, sigma_bs=sigma_bs, call_put=pinn.call_put
    )
    beta = float(getattr(pinn, "price_softplus_beta", 0.0))
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
    N_points=2000,
    sigma_mode="spot_var",
    stationary=True,
    lambda_price_low=1.0,
    lambda_delta_low=0.0,
    lambda_price_high=1.0,
    lambda_delta_high=0.1,
):
    """Asymptotic boundary conditions in log-moneyness x."""
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    _, X, tau, r, xi0, omega, kappa, rho = _sample_interior(pinn, N_points)

    x_low = torch.full(
        (N_points, 1), pinn.x_min, device=device, dtype=dtype
    ).requires_grad_(True)
    x_high = torch.full(
        (N_points, 1), pinn.x_max, device=device, dtype=dtype
    ).requires_grad_(True)

    u_low, _, _ = _total_price(
        pinn, x_low, X, tau, r, xi0, omega, kappa, rho, sigma_mode, stationary
    )
    u_high, _, _ = _total_price(
        pinn, x_high, X, tau, r, xi0, omega, kappa, rho, sigma_mode, stationary
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


# ============================================================
# PINN model
# ============================================================


class PINN(nn.Module):
    """
    PINN for normalized 1-factor Bergomi correction:

        u_bergomi(x, X, ...) ~= u_BS(x, ...) + U(x, X, ...)

    with x = log(S / K), u = V / K.
    """

    def __init__(
        self,
        x_min,
        x_max,
        X_max,
        T,
        r_max,
        xi0_max,
        omega_max,
        kappa_max,
        call_put="Call",
        hidden=128,
        depth=4,
        v_max=1.0,
        kappa_floor=0.25,
        price_softplus_beta=0.0,
    ):
        super().__init__()

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.x_scale = max(abs(x_min), abs(x_max), 1e-6)

        self.X_max = float(X_max)
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

        input_dim = 22
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

    def _features_from_x(self, x, X, tau, r, xi0, omega, kappa, rho):
        x_norm = x / self.x_scale
        X_norm = X / self.X_max
        tau_norm = tau / self.T
        r_norm = r / self.r_max
        xi0_norm = xi0 / self.xi0_max
        omega_norm = omega / self.omega_max
        kappa_norm = kappa / self.kappa_max
        rho_norm = rho

        tau_sq = tau_norm ** 2
        tau_sin = torch.sin(torch.pi * tau_norm)
        tau_cos = torch.cos(torch.pi * tau_norm)

        # Stationary log-variance drive (parameter-free feature of Bergomi)
        log_v_drive = omega * X - (omega ** 2) / (4.0 * kappa.clamp_min(1e-6))
        log_v_drive_norm = log_v_drive / (1.0 + self.omega_max * self.X_max)

        mr_clock = kappa * tau
        mr_decay = torch.exp(-torch.clamp(mr_clock, min=0.0, max=50.0))
        mr_strength = 1.0 - mr_decay

        x_sq = x_norm ** 2
        x_cube = x_norm ** 3
        abs_x = torch.abs(x_norm)
        X_sq = X_norm ** 2
        abs_X = torch.abs(X_norm)

        x_tau = x_norm * tau_norm
        x_X = x_norm * X_norm
        X_tau = X_norm * tau_norm

        cp_feat = self._cp_feature(x)

        return torch.cat(
            [
                x_norm,
                X_norm,
                tau_norm,
                r_norm,
                xi0_norm,
                omega_norm,
                kappa_norm,
                rho_norm,
                tau_sq,
                tau_sin,
                tau_cos,
                log_v_drive_norm,
                mr_strength,
                x_sq,
                x_cube,
                abs_x,
                X_sq,
                abs_X,
                x_tau,
                x_X,
                X_tau,
                cp_feat,
            ],
            dim=1,
        )

    def forward_x(self, x, X, tau, r, xi0, omega, kappa, rho):
        features = self._features_from_x(x, X, tau, r, xi0, omega, kappa, rho)
        raw = self.net(features)
        # Keep U small vs the BS baseline. Gate by omega so U -> 0 as omega -> 0
        # (Bergomi collapses to Black–Scholes when vol-of-variance vanishes).
        omega_gate = (omega / max(self.omega_max, 1e-6)).clamp(0.0, 1.0)
        corr_scale = 0.15 * (0.5 + tau / self.T) * (1.0 + 0.25 * torch.abs(x))
        return omega_gate * corr_scale * torch.tanh(raw)

    def forward(self, S, K, X, tau, r, xi0, omega, kappa, rho):
        eps = 1e-8
        x = torch.log(torch.clamp(S, min=eps) / torch.clamp(K, min=eps))
        return self.forward_x(x, X, tau, r, xi0, omega, kappa, rho)

    @staticmethod
    def _to_col(z, device):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32)
        if z.dim() == 0:
            z = z.view(1, 1)
        elif z.dim() == 1:
            z = z.view(-1, 1)
        return z.to(device)

    def predict_correction(self, S, K, X, tau, r, xi0, omega, kappa, rho):
        device = next(self.parameters()).device
        S, K, X, tau, r, xi0, omega, kappa, rho = [
            self._to_col(z, device)
            for z in (S, K, X, tau, r, xi0, omega, kappa, rho)
        ]
        with torch.no_grad():
            return self.forward(S, K, X, tau, r, xi0, omega, kappa, rho)

    def predict_normalized_price(
        self,
        S,
        K,
        X,
        tau,
        r,
        xi0,
        omega,
        kappa,
        rho,
        sigma_mode="spot_var",
        stationary=True,
    ):
        device = next(self.parameters()).device
        S_t = self._to_col(S, device)
        K_t = self._to_col(K, device)
        X_t = self._to_col(X, device)
        tau_t = self._to_col(tau, device)
        r_t = self._to_col(r, device)
        xi0_t = self._to_col(xi0, device)
        omega_t = self._to_col(omega, device)
        kappa_t = self._to_col(kappa, device)
        rho_t = self._to_col(rho, device)

        eps = 1e-8
        x_t = torch.log(torch.clamp(S_t, min=eps) / torch.clamp(K_t, min=eps))

        with torch.no_grad():
            U = self.forward_x(
                x_t, X_t, tau_t, r_t, xi0_t, omega_t, kappa_t, rho_t
            )
            t = None if stationary else (self.T - tau_t).clamp_min(0.0)
            v = v_bergomi(
                X_t,
                xi0_t,
                omega_t,
                kappa_t,
                t=t,
                stationary=stationary,
                v_max=getattr(self, "v_max", None),
            )
            sigma_bs = sigma_bs_bergomi(
                xi0=xi0_t, v=v, tau=tau_t, mode=sigma_mode
            )
            u_bs = bs_option_normalized_from_x(
                x=x_t,
                tau=tau_t,
                r=r_t,
                sigma_bs=sigma_bs,
                call_put=self.call_put,
            )
            beta = float(getattr(self, "price_softplus_beta", 0.0))
            return _soft_positive_price(u_bs + U, beta=beta)

    def predict_price(
        self,
        S,
        K,
        X,
        tau,
        r,
        xi0,
        omega,
        kappa,
        rho,
        sigma_mode="spot_var",
        stationary=True,
    ):
        device = next(self.parameters()).device
        K_t = self._to_col(K, device)
        u_hat = self.predict_normalized_price(
            S=S,
            K=K,
            X=X,
            tau=tau,
            r=r,
            xi0=xi0,
            omega=omega,
            kappa=kappa,
            rho=rho,
            sigma_mode=sigma_mode,
            stationary=stationary,
        )
        return K_t * u_hat


# ============================================================
# Supervised (hybrid) loss helpers
# ============================================================


def _move_supervised_batch(batch, device, dtype):
    """Copy a supervised minibatch dict onto device/dtype."""
    out = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device=device, dtype=dtype)
        else:
            out[key] = val
    return out


def sample_supervised_batch(supervised_data, N_data, device, dtype):
    """Draw a random minibatch from a precomputed QMC label dataset."""
    if "u_target" in supervised_data:
        anchor = supervised_data["u_target"]
    elif "delta_target" in supervised_data:
        anchor = supervised_data["delta_target"]
    else:
        raise ValueError(
            "supervised_data must contain 'u_target' or 'delta_target'"
        )
    n_total = int(anchor.shape[0])
    if n_total <= 0:
        raise ValueError("supervised_data is empty")
    n = min(int(N_data), n_total)
    idx = torch.randint(0, n_total, (n,), device=anchor.device)
    batch = {
        k: v[idx] for k, v in supervised_data.items() if isinstance(v, torch.Tensor)
    }
    return _move_supervised_batch(batch, device, dtype)


def supervised_price_loss(
    pinn,
    batch,
    sigma_mode="spot_var",
    stationary=True,
    relative: bool = False,
):
    """
    Weighted MSE between PINN normalized price and QMC+ADD labels.

    ``batch`` must contain tensors: x, X, tau, r, xi0, omega, kappa, rho,
    u_target, and optional weight.

    Default is absolute error in normalized price (ATM weights in the
    dataset already temper ITM dominance). Set ``relative=True`` to divide
    by ``1+|u_BS|`` (can starve ITM gradients).
    """
    x = batch["x"]
    X = batch["X"]
    tau = batch["tau"]
    r = batch["r"]
    xi0 = batch["xi0"]
    omega = batch["omega"]
    kappa = batch["kappa"]
    rho = batch["rho"]
    u_target = batch["u_target"]
    weight = batch.get("weight")
    if weight is None:
        weight = torch.ones_like(u_target)

    u_pred, u_bs, _ = _total_price(
        pinn, x, X, tau, r, xi0, omega, kappa, rho, sigma_mode, stationary
    )
    err = u_pred - u_target
    if relative:
        err = err / (1.0 + torch.abs(u_bs.detach()))
    return torch.mean(weight * err**2)


def supervised_price_l1(
    pinn,
    batch,
    sigma_mode="spot_var",
    stationary=True,
):
    """Mean absolute normalized-price error on a labeled batch (no grad)."""
    x = batch["x"]
    X = batch["X"]
    tau = batch["tau"]
    r = batch["r"]
    xi0 = batch["xi0"]
    omega = batch["omega"]
    kappa = batch["kappa"]
    rho = batch["rho"]
    u_target = batch["u_target"]
    u_pred, _, _ = _total_price(
        pinn, x, X, tau, r, xi0, omega, kappa, rho, sigma_mode, stationary
    )
    return torch.mean(torch.abs(u_pred - u_target))


def supervised_delta_loss(
    pinn,
    batch,
    sigma_mode="spot_var",
    stationary=True,
):
    """
    MSE on pathwise Delta labels.

    ``batch`` needs the usual market features plus ``delta_target``.
    Delta = ∂V/∂S with V = K·u and x = log(S/K) ⇒ Δ = e^{-x} u_x.
    """
    x = batch["x"].detach().requires_grad_(True)
    X = batch["X"]
    tau = batch["tau"]
    r = batch["r"]
    xi0 = batch["xi0"]
    omega = batch["omega"]
    kappa = batch["kappa"]
    rho = batch["rho"]
    delta_target = batch["delta_target"]
    weight = batch.get("weight")
    if weight is None:
        weight = torch.ones_like(delta_target)

    u_pred, _, _ = _total_price(
        pinn, x, X, tau, r, xi0, omega, kappa, rho, sigma_mode, stationary
    )
    u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
    delta_pred = torch.exp(-x) * u_x
    err = delta_pred - delta_target
    return torch.mean(weight * err**2)


def otm_positivity_loss(
    pinn,
    N_points=2000,
    sigma_mode="spot_var",
    stationary=True,
    otm_fraction=0.7,
):
    """
    Soft hinge that penalizes negative normalized call prices.

    Oversamples deep OTM log-moneyness so the wing that produced extra
    negatives under hybrid fine-tuning is directly regularized.
    """
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype
    n_otm = int(round(float(otm_fraction) * N_points))
    n_rest = N_points - n_otm

    x_rest, X, tau, r, xi0, omega, kappa, rho = _sample_interior(pinn, n_rest)
    # Deep OTM for calls: x in [x_min, ~-0.25]
    x_hi = min(float(pinn.x_max), -0.25)
    x_lo = float(pinn.x_min)
    x_otm = (
        x_lo + (x_hi - x_lo) * torch.rand(n_otm, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    # Reuse factor/param draws from the rest sample by resampling interior for OTM
    # size (keeps X~stationary OU coupling intact).
    x_fill, X_otm, tau_otm, r_otm, xi0_otm, omega_otm, kappa_otm, rho_otm = (
        _sample_interior(pinn, n_otm)
    )
    _ = x_fill
    x = torch.cat([x_rest, x_otm], dim=0)
    X = torch.cat([X, X_otm], dim=0)
    tau = torch.cat([tau, tau_otm], dim=0)
    r = torch.cat([r, r_otm], dim=0)
    xi0 = torch.cat([xi0, xi0_otm], dim=0)
    omega = torch.cat([omega, omega_otm], dim=0)
    kappa = torch.cat([kappa, kappa_otm], dim=0)
    rho = torch.cat([rho, rho_otm], dim=0)

    u, _, _ = _total_price(
        pinn, x, X, tau, r, xi0, omega, kappa, rho, sigma_mode, stationary
    )
    return torch.mean(torch.nn.functional.relu(-u) ** 2)


# ============================================================
# Training
# ============================================================


def train_network(
    pinn,
    N_pde=5000,
    N_boundary=2000,
    epochs=5000,
    lr=1e-4,
    sigma_mode="spot_var",
    stationary=True,
    lambda_boundary=1.0,
    lambda_terminal=1.0,
    lambda_data=0.0,
    lambda_delta=0.0,
    lambda_otm=0.0,
    supervised_data=None,
    supervised_delta_data=None,
    val_data=None,
    N_data=256,
    N_delta=64,
    N_otm=2000,
    detach_source=False,
    grad_clip=1.0,
    print_every=100,
    best_model_path="best_pinn_bergomi_xspace.pt",
    save_model=True,
    weight_decay=1e-2,
    cosine_eta_min=1e-5,
    cosine_T0=3000,
    detect_anomaly=False,
    extra_checkpoint_meta=None,
    selection_metric="auto",
    early_stop_patience=None,
    min_epochs_for_early_stop=0,
):
    """
    Train Bergomi PINN.

    ``selection_metric``:
      - ``"auto"``: val L1 if ``val_data`` given, else full data loss if
        supervised, else total loss
      - ``"val_l1"`` / ``"data"`` / ``"total"``
    """
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    use_data = (
        supervised_data is not None
        and float(lambda_data) > 0.0
        and int(N_data) > 0
    )
    use_delta = (
        supervised_delta_data is not None
        and float(lambda_delta) > 0.0
        and int(N_delta) > 0
    )
    use_otm = float(lambda_otm) > 0.0 and int(N_otm) > 0
    use_val = val_data is not None

    if supervised_data is not None:
        supervised_data = _move_supervised_batch(supervised_data, device, dtype)
    if supervised_delta_data is not None:
        supervised_delta_data = _move_supervised_batch(
            supervised_delta_data, device, dtype
        )
    if val_data is not None:
        val_data = _move_supervised_batch(val_data, device, dtype)

    if selection_metric == "auto":
        if use_val:
            selection_metric = "val_l1"
        elif use_data:
            selection_metric = "data"
        else:
            selection_metric = "total"

    optimizer = torch.optim.AdamW(
        pinn.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(cosine_T0),
        eta_min=cosine_eta_min,
    )

    history = {
        "epoch": [],
        "elapsed_time": [],
        "total": [],
        "physics": [],
        "terminal": [],
        "boundary": [],
        "data": [],
        "delta": [],
        "otm": [],
        "val_l1": [],
        "lr": [],
        "model depth": pinn.depth,
        "model width": pinn.hidden,
        "sigma_mode": sigma_mode,
        "stationary": stationary,
        "lambda_data": float(lambda_data),
        "lambda_delta": float(lambda_delta),
        "lambda_otm": float(lambda_otm),
        "selection_metric": selection_metric,
        "N_data": int(N_data) if use_data else 0,
        "N_delta": int(N_delta) if use_delta else 0,
        "n_supervised": (
            int(supervised_data["u_target"].shape[0]) if use_data else 0
        ),
        "n_delta": (
            int(supervised_delta_data["delta_target"].shape[0])
            if use_delta
            else 0
        ),
        "n_val": int(val_data["u_target"].shape[0]) if use_val else 0,
    }

    best_loss = float("inf")
    best_epoch = None
    best_state_dict = None
    best_data_loss = float("inf")
    best_val_l1 = float("inf")
    best_metric = float("inf")
    epochs_since_improve = 0
    start_time = time.time()

    torch.autograd.set_detect_anomaly(detect_anomaly)

    def _full_data_loss():
        if not use_data:
            return float("inf")
        pinn.eval()
        with torch.no_grad():
            loss_val = supervised_price_loss(
                pinn,
                supervised_data,
                sigma_mode=sigma_mode,
                stationary=stationary,
            )
        pinn.train()
        return float(loss_val.item())

    def _val_l1():
        if not use_val:
            return float("inf")
        pinn.eval()
        with torch.no_grad():
            l1 = supervised_price_l1(
                pinn,
                val_data,
                sigma_mode=sigma_mode,
                stationary=stationary,
            )
        pinn.train()
        return float(l1.item())

    for epoch in range(1, epochs + 1):
        pinn.train()
        optimizer.zero_grad(set_to_none=True)

        loss_physics = pde_dynamic_x(
            pinn=pinn,
            N_points=N_pde,
            sigma_mode=sigma_mode,
            stationary=stationary,
            detach_source=detach_source,
        )
        loss_terminal = spot_terminal_condition_x(
            pinn=pinn,
            N_points=N_boundary,
            sigma_mode=sigma_mode,
            stationary=stationary,
        )
        loss_boundary = spot_boundary_conditions_x(
            pinn=pinn,
            N_points=N_boundary,
            sigma_mode=sigma_mode,
            stationary=stationary,
            lambda_price_low=1.0,
            lambda_delta_low=0.05 if use_otm else 0.0,
            lambda_price_high=1.0,
            lambda_delta_high=0.1,
        )

        if use_data:
            batch = sample_supervised_batch(
                supervised_data, N_data=N_data, device=device, dtype=dtype
            )
            loss_data = supervised_price_loss(
                pinn,
                batch,
                sigma_mode=sigma_mode,
                stationary=stationary,
            )
        else:
            loss_data = torch.zeros((), device=device, dtype=dtype)

        if use_delta:
            dbatch = sample_supervised_batch(
                supervised_delta_data,
                N_data=N_delta,
                device=device,
                dtype=dtype,
            )
            loss_delta = supervised_delta_loss(
                pinn,
                dbatch,
                sigma_mode=sigma_mode,
                stationary=stationary,
            )
        else:
            loss_delta = torch.zeros((), device=device, dtype=dtype)

        if use_otm:
            loss_otm = otm_positivity_loss(
                pinn,
                N_points=N_otm,
                sigma_mode=sigma_mode,
                stationary=stationary,
            )
        else:
            loss_otm = torch.zeros((), device=device, dtype=dtype)

        loss = (
            loss_physics
            + lambda_terminal * loss_terminal
            + lambda_boundary * loss_boundary
            + float(lambda_data) * loss_data
            + float(lambda_delta) * loss_delta
            + float(lambda_otm) * loss_otm
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite total loss at epoch {epoch}: "
                f"physics={loss_physics.item()}, "
                f"terminal={loss_terminal.item()}, "
                f"boundary={loss_boundary.item()}, "
                f"data={float(loss_data.item())}, "
                f"delta={float(loss_delta.item())}, "
                f"otm={float(loss_otm.item())}"
            )

        loss.backward()

        for name, param in pinn.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient at epoch {epoch} param={name}"
                )

        grad_norm = None
        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(pinn.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        for name, param in pinn.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(
                    f"Non-finite parameter after step at epoch {epoch} param={name}"
                )

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        current_loss = loss.item()
        data_val = float(loss_data.item())
        delta_val = float(loss_delta.item())
        otm_val = float(loss_otm.item())

        history["epoch"].append(epoch)
        history["elapsed_time"].append(elapsed)
        history["total"].append(current_loss)
        history["physics"].append(loss_physics.item())
        history["terminal"].append(loss_terminal.item())
        history["boundary"].append(loss_boundary.item())
        history["data"].append(data_val)
        history["delta"].append(delta_val)
        history["otm"].append(otm_val)
        history["lr"].append(current_lr)

        improved = False
        do_eval = (
            epoch == 1
            or epoch % print_every == 0
            or epoch == epochs
            or selection_metric == "total"
        )
        metric_now = None
        val_l1_now = None
        full_data = None

        if do_eval or selection_metric in ("val_l1", "data"):
            if use_val and (
                selection_metric == "val_l1"
                or epoch == 1
                or epoch % print_every == 0
                or epoch == epochs
            ):
                val_l1_now = _val_l1()
                history["val_l1"].append({"epoch": epoch, "val_l1": val_l1_now})
            if use_data and (
                selection_metric == "data"
                or epoch == 1
                or epoch % print_every == 0
                or epoch == epochs
            ):
                full_data = _full_data_loss()
                history.setdefault("data_full", []).append(
                    {"epoch": epoch, "data_full": full_data}
                )

            if selection_metric == "val_l1":
                metric_now = val_l1_now
            elif selection_metric == "data":
                metric_now = full_data
            else:
                metric_now = current_loss

            if metric_now is not None and metric_now < best_metric:
                best_metric = metric_now
                best_loss = current_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(pinn.state_dict())
                if full_data is not None:
                    best_data_loss = full_data
                if val_l1_now is not None:
                    best_val_l1 = val_l1_now
                improved = True
                epochs_since_improve = 0
            elif selection_metric != "total" and (
                epoch == 1 or epoch % print_every == 0 or epoch == epochs
            ):
                epochs_since_improve += 1

        if selection_metric == "total" and current_loss < best_metric:
            best_metric = current_loss
            best_loss = current_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(pinn.state_dict())
            improved = True
            epochs_since_improve = 0

        if improved and save_model:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": best_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "data_full_loss": (
                    best_data_loss if use_data else None
                ),
                "val_l1": best_val_l1 if use_val else None,
                "selection_metric": selection_metric,
                "best_metric": best_metric,
                "sigma_mode": sigma_mode,
                "stationary": stationary,
                "call_put": getattr(pinn, "call_put", None),
                "x_min": getattr(pinn, "x_min", None),
                "x_max": getattr(pinn, "x_max", None),
                "X_max": getattr(pinn, "X_max", None),
                "T": getattr(pinn, "T", None),
                "r_max": getattr(pinn, "r_max", None),
                "xi0_max": getattr(pinn, "xi0_max", None),
                "omega_max": getattr(pinn, "omega_max", None),
                "kappa_max": getattr(pinn, "kappa_max", None),
                "v_max": getattr(pinn, "v_max", None),
                "kappa_floor": getattr(pinn, "kappa_floor", None),
                "price_softplus_beta": getattr(
                    pinn, "price_softplus_beta", 0.0
                ),
                "hidden": pinn.hidden,
                "depth": pinn.depth,
                "lambda_data": float(lambda_data),
                "lambda_delta": float(lambda_delta),
                "lambda_otm": float(lambda_otm),
                "hybrid_supervised": bool(use_data or use_delta),
            }
            if extra_checkpoint_meta:
                ckpt.update(extra_checkpoint_meta)
            torch.save(ckpt, best_model_path)

        if epoch % print_every == 0 or epoch == 1:
            grad_norm_str = (
                f"{float(grad_norm):.3e}" if grad_norm is not None else "None"
            )
            if selection_metric == "val_l1":
                best_str = f"BestValL1: {best_val_l1:.6e} @ {best_epoch}"
            elif selection_metric == "data":
                best_str = f"BestData: {best_data_loss:.6e} @ {best_epoch}"
            else:
                best_str = f"Best: {best_loss:.6e} @ {best_epoch}"
            val_str = (
                f"ValL1: {val_l1_now:.6e} | "
                if val_l1_now is not None
                else ""
            )
            print(
                f"Epoch {epoch:6d} | "
                f"Time: {elapsed:10.2f}s | "
                f"LR: {current_lr:.3e} | "
                f"Total: {current_loss:.6e} | "
                f"PDE_x: {loss_physics.item():.6e} | "
                f"Terminal_x: {loss_terminal.item():.6e} | "
                f"Boundary_x: {loss_boundary.item():.6e} | "
                f"Data: {data_val:.6e} | "
                f"Delta: {delta_val:.6e} | "
                f"OTM: {otm_val:.6e} | "
                f"{val_str}"
                f"GradNorm: {grad_norm_str} | "
                f"{best_str}"
            )

        if (
            early_stop_patience is not None
            and selection_metric != "total"
            and epoch >= int(min_epochs_for_early_stop)
            and epochs_since_improve >= int(early_stop_patience)
        ):
            print(
                f"Early stop at epoch {epoch}: no {selection_metric} "
                f"improvement for {early_stop_patience} evals "
                f"(best @ {best_epoch})",
                flush=True,
            )
            break

    history["best_loss"] = best_loss
    history["best_epoch"] = best_epoch
    history["best_data_loss"] = best_data_loss if use_data else None
    history["best_val_l1"] = best_val_l1 if use_val else None
    history["best_metric"] = best_metric

    if best_state_dict is not None:
        pinn.load_state_dict(best_state_dict)

    return history
