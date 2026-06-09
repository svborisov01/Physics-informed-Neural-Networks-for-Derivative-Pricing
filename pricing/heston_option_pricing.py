import torch
import torch.nn as nn
import numpy as np
import time
import copy

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from support_tools.analytical_pricing_tools import bs_option_normalized_from_x, sigma_bs_effective

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def pde_dynamic_x(pinn, N_points=5000, sigma_mode="hybrid", detach_source=False):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    x = (
        pinn.x_min + (pinn.x_max - pinn.x_min)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    v = (
        1e-4 + (pinn.v_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    tau = (
        1e-4 + (pinn.T - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    r = (
        1e-4 + (pinn.r_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    kappa = (
        1e-4 + (pinn.kappa_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    theta = (
        1e-4 + (pinn.theta_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    sigma = (
        1e-4 + (pinn.sigma_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    rho = (
        -0.95 + 1.90 * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    def heston_operator(f):
        f_tau = torch.autograd.grad(f.sum(), tau, create_graph=True)[0]
        f_x   = torch.autograd.grad(f.sum(), x,   create_graph=True)[0]
        f_v   = torch.autograd.grad(f.sum(), v,   create_graph=True)[0]

        f_xx  = torch.autograd.grad(f_x.sum(), x, create_graph=True)[0]
        f_vv  = torch.autograd.grad(f_v.sum(), v, create_graph=True)[0]
        f_xv  = torch.autograd.grad(f_x.sum(), v, create_graph=True)[0]

        return (
            -f_tau
            + (r - 0.5 * v) * f_x
            + 0.5 * v * f_xx
            + rho * sigma * v * f_xv
            + 0.5 * sigma**2 * v * f_vv
            + kappa * (theta - v) * f_v
            - r * f
        )

    # Learnable correction U(x, v, tau, r, kappa, theta, sigma, rho)
    U = pinn.forward_x(x, v, tau, r, kappa, theta, sigma, rho)

    # Black-Scholes baseline u_bs(x, tau, r; sigma_eff)
    sigma_bs = sigma_bs_effective(
        v=v,
        theta=theta,
        kappa=kappa,
        tau=tau,
        mode=sigma_mode
    )

    u_bs = bs_option_normalized_from_x(
        x=x,
        tau=tau,
        r=r,
        sigma_bs=sigma_bs,
        call_put=pinn.call_put
    )

    if detach_source:
        u_bs = u_bs.detach()

    # Total normalized price
    u_total = u_bs + U

    # Enforce Heston PDE on the total price, not separately on U and u_bs
    residual = heston_operator(u_total)

    # Scale by baseline magnitude for stability
    residual = residual / (1.0 + torch.abs(u_bs))

    return torch.mean(residual ** 2)

def spot_terminal_condition_x(pinn, N_points=2000, sigma_mode="hybrid"):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    x = (
        pinn.x_min + (pinn.x_max - pinn.x_min)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    v = (
        1e-4 + (pinn.v_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    tau = torch.zeros(N_points, 1, device=device, dtype=dtype)

    r = (
        1e-4 + (pinn.r_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    kappa = (
        1e-4 + (pinn.kappa_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    theta = (
        1e-4 + (pinn.theta_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    sigma = (
        1e-4 + (pinn.sigma_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    rho = (
        -0.95 + 1.90 * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    # Learnable correction
    U = pinn.forward_x(x, v, tau, r, kappa, theta, sigma, rho)

    # BS baseline near maturity; use tiny positive tau for numerical stability
    tau_safe = torch.full_like(x, 1e-6)

    sigma_bs = sigma_bs_effective(
        v=v,
        theta=theta,
        kappa=kappa,
        tau=tau_safe,
        mode=sigma_mode
    )

    u_bs = bs_option_normalized_from_x(
        x=x,
        tau=tau_safe,
        r=r,
        sigma_bs=sigma_bs,
        call_put=pinn.call_put
    )

    # Total normalized price
    u_total = u_bs + U

    cp = pinn.call_put.lower()
    if cp == "call":
        payoff = torch.relu(torch.exp(x) - 1.0)
    elif cp == "put":
        payoff = torch.relu(1.0 - torch.exp(x))
    else:
        raise ValueError("pinn.call_put must be either 'Call' or 'Put'")

    # Optional normalization for stability
    return torch.mean(((u_total - payoff) / (1.0 + payoff)) ** 2)

def spot_boundary_conditions_x(
    pinn,
    N_points=2000,
    sigma_mode="hybrid",
    lambda_price_low=1.0,
    lambda_delta_low=0.0,
    lambda_price_high=1.0,
    lambda_delta_high=0.1,
):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    v = (
        1e-4 + (pinn.v_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    tau = (
        1e-4 + (pinn.T - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    r = (
        1e-4 + (pinn.r_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    kappa = (
        1e-4 + (pinn.kappa_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    theta = (
        1e-4 + (pinn.theta_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    sigma = (
        1e-4 + (pinn.sigma_max - 1e-4)
        * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    rho = (
        -0.95 + 1.90 * torch.rand(N_points, 1, device=device, dtype=dtype)
    ).requires_grad_(True)

    x_low = torch.full(
        (N_points, 1), pinn.x_min, device=device, dtype=dtype
    ).requires_grad_(True)

    x_high = torch.full(
        (N_points, 1), pinn.x_max, device=device, dtype=dtype
    ).requires_grad_(True)

    sigma_bs = sigma_bs_effective(
        v=v,
        theta=theta,
        kappa=kappa,
        tau=tau,
        mode=sigma_mode
    )

    cp = pinn.call_put.lower()

    # ----- lower boundary -----
    U_low = pinn.forward_x(x_low, v, tau, r, kappa, theta, sigma, rho)
    u_bs_low = bs_option_normalized_from_x(
        x=x_low,
        tau=tau,
        r=r,
        sigma_bs=sigma_bs,
        call_put=pinn.call_put
    )
    u_low = u_bs_low + U_low
    ux_low = torch.autograd.grad(u_low.sum(), x_low, create_graph=True)[0]

    if cp == "call":
        # x -> -inf: call price -> 0, delta_x -> 0
        target_price_low = torch.zeros_like(u_low)
        target_delta_low = torch.zeros_like(ux_low)

    elif cp == "put":
        # x -> -inf: normalized put price -> exp(-r tau), delta_x -> 0
        target_price_low = torch.exp(-r * tau)
        target_delta_low = torch.zeros_like(ux_low)

    else:
        raise ValueError("pinn.call_put must be either 'Call' or 'Put'")

    # ----- upper boundary -----
    U_high = pinn.forward_x(x_high, v, tau, r, kappa, theta, sigma, rho)
    u_bs_high = bs_option_normalized_from_x(
        x=x_high,
        tau=tau,
        r=r,
        sigma_bs=sigma_bs,
        call_put=pinn.call_put
    )
    u_high = u_bs_high + U_high
    ux_high = torch.autograd.grad(u_high.sum(), x_high, create_graph=True)[0]

    if cp == "call":
        # x -> +inf: normalized call price -> exp(x) - exp(-r tau), delta_x -> exp(x)
        target_price_high = torch.exp(x_high) - torch.exp(-r * tau)
        target_delta_high = torch.exp(x_high)

    elif cp == "put":
        # x -> +inf: normalized put price -> 0, delta_x -> 0
        target_price_high = torch.zeros_like(u_high)
        target_delta_high = torch.zeros_like(ux_high)

    # normalized errors for better stability
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
    PINN for normalized Heston correction:
        u_heston(x,...) ~= u_bs(x,...) + U(x,...)

    where
        x = log(S / K),
        u = V / K

    The network learns only the correction U.
    The total price is constructed outside the network as u_bs + U.
    """

    def __init__(
        self,
        x_min,
        x_max,
        v_max,
        T,
        r_max,
        kappa_max,
        theta_max,
        sigma_max,
        call_put="Call",
        hidden=128,
        depth=4,
    ):
        super().__init__()

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.x_scale = max(abs(x_min), abs(x_max), 1e-6)

        self.v_max = float(v_max)
        self.T = float(T)
        self.r_max = float(r_max)
        self.kappa_max = float(kappa_max)
        self.theta_max = float(theta_max)
        self.sigma_max = float(sigma_max)
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
        elif cp == "put":
            return -torch.ones_like(x)
        else:
            raise ValueError("call_put must be either 'Call' or 'Put'")

    def _features_from_x(self, x, v, tau, r, kappa, theta, sigma, rho):
        x_norm = x / self.x_scale
        v_norm = v / self.v_max
        tau_norm = tau / self.T
        r_norm = r / self.r_max
        kappa_norm = kappa / self.kappa_max
        theta_norm = theta / self.theta_max
        sigma_norm = sigma / self.sigma_max
        rho_norm = rho

        tau_sq = tau_norm ** 2
        tau_sin = torch.sin(torch.pi * tau_norm)
        tau_cos = torch.cos(torch.pi * tau_norm)

        denom_vtheta = max(self.v_max + self.theta_max, 1e-6)
        gap = v - theta
        gap_norm = gap / denom_vtheta
        abs_gap = torch.abs(gap_norm)

        mr_clock = kappa * tau
        mr_decay = torch.exp(-torch.clamp(mr_clock, min=0.0, max=50.0))
        mr_strength = 1.0 - mr_decay
        gap_effect = gap_norm * mr_strength

        # Symmetric-in-x basis around ATM
        x_sq = x_norm ** 2
        x_cube = x_norm ** 3
        abs_x = torch.abs(x_norm)

        # Mild interaction features
        x_tau = x_norm * tau_norm
        x_v = x_norm * v_norm
        x_gap = x_norm * gap_norm

        # Call/put indicator
        cp_feat = self._cp_feature(x)

        return torch.cat([
            x_norm,
            v_norm,
            tau_norm,
            r_norm,
            kappa_norm,
            theta_norm,
            sigma_norm,
            rho_norm,
            tau_sq,
            tau_sin,
            tau_cos,
            gap_norm,
            abs_gap,
            mr_strength,
            gap_effect,
            x_sq,
            x_cube,
            abs_x,
            x_tau,
            cp_feat,
            x_v,
            x_gap
        ], dim=1)

    def forward_x(self, x, v, tau, r, kappa, theta, sigma, rho):
        features = self._features_from_x(x, v, tau, r, kappa, theta, sigma, rho)
        raw = self.net(features)

        # Aggresive regularization helps to solve mispricing for high S and tau
        corr_scale = 2.0 * (1.0 + torch.abs(x)) * (1.0 + tau / self.T)
        U = corr_scale * torch.tanh(raw)

        return U

    def forward(self, S, K, v, tau, r, kappa, theta, sigma, rho):
        eps = 1e-8
        x = torch.log(torch.clamp(S, min=eps) / torch.clamp(K, min=eps))
        return self.forward_x(x, v, tau, r, kappa, theta, sigma, rho)

    def predict_correction(self, S, K, v, tau, r, kappa, theta, sigma, rho):
        def _to_tensor(z):
            if not isinstance(z, torch.Tensor):
                z = torch.tensor(z, dtype=torch.float32)
            if z.dim() == 0:
                return z.view(1, 1)
            elif z.dim() == 1:
                return z.view(-1, 1)
            return z

        S, K, v, tau, r, kappa, theta, sigma, rho = [
            _to_tensor(z) for z in [S, K, v, tau, r, kappa, theta, sigma, rho]
        ]

        device = next(self.parameters()).device
        S, K, v, tau, r, kappa, theta, sigma, rho = [
            z.to(device) for z in [S, K, v, tau, r, kappa, theta, sigma, rho]
        ]

        with torch.no_grad():
            return self.forward(S, K, v, tau, r, kappa, theta, sigma, rho)

    def predict_normalized_price(self, S, K, v, tau, r, kappa, theta, sigma, rho, sigma_mode="hybrid"):
        S_t = torch.as_tensor(S, dtype=torch.float32).reshape(-1, 1)
        K_t = torch.as_tensor(K, dtype=torch.float32).reshape(-1, 1)
        v_t = torch.as_tensor(v, dtype=torch.float32).reshape(-1, 1)
        tau_t = torch.as_tensor(tau, dtype=torch.float32).reshape(-1, 1)
        r_t = torch.as_tensor(r, dtype=torch.float32).reshape(-1, 1)
        kappa_t = torch.as_tensor(kappa, dtype=torch.float32).reshape(-1, 1)
        theta_t = torch.as_tensor(theta, dtype=torch.float32).reshape(-1, 1)
        sigma_t = torch.as_tensor(sigma, dtype=torch.float32).reshape(-1, 1)
        rho_t = torch.as_tensor(rho, dtype=torch.float32).reshape(-1, 1)

        device = next(self.parameters()).device
        S_t, K_t, v_t, tau_t, r_t, kappa_t, theta_t, sigma_t, rho_t = [
            z.to(device) for z in [S_t, K_t, v_t, tau_t, r_t, kappa_t, theta_t, sigma_t, rho_t]
        ]

        eps = 1e-8
        x_t = torch.log(torch.clamp(S_t, min=eps) / torch.clamp(K_t, min=eps))

        with torch.no_grad():
            U = self.forward_x(x_t, v_t, tau_t, r_t, kappa_t, theta_t, sigma_t, rho_t)

            sigma_bs = sigma_bs_effective(
                v=v_t,
                theta=theta_t,
                kappa=kappa_t,
                tau=tau_t,
                mode=sigma_mode
            )

            u_bs = bs_option_normalized_from_x(
                x=x_t,
                tau=tau_t,
                r=r_t,
                sigma_bs=sigma_bs,
                call_put=self.call_put
            )

            u_hat = u_bs + U

        return u_hat

    def predict_price(self, S, K, v, tau, r, kappa, theta, sigma, rho, sigma_mode="hybrid"):
        K_t = torch.as_tensor(K, dtype=torch.float32).reshape(-1, 1)
        device = next(self.parameters()).device
        K_t = K_t.to(device)

        u_hat = self.predict_normalized_price(
            S=S, K=K, v=v, tau=tau, r=r,
            kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            sigma_mode=sigma_mode
        )

        return K_t * u_hat

def train_network(
    pinn,
    N_pde=5000,
    N_boundary=2000,
    epochs=5000,
    lr=1e-4,
    sigma_mode="hybrid",
    lambda_boundary=1.0,
    lambda_terminal=0.5,
    grad_clip=1.0,
    print_every=100,
    best_model_path="best_pinn_heston_xspace.pt",
    save_model = True,
    weight_decay=1e-2,
    cosine_eta_min=1e-5,
    detect_anomaly=False,
):
    device = next(pinn.parameters()).device

    optimizer = torch.optim.AdamW(
        pinn.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=3000,
        eta_min=cosine_eta_min,
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
        "model width": pinn.hidden
    }

    best_loss = float("inf")
    best_epoch = None
    best_state_dict = None
    start_time = time.time()

    torch.autograd.set_detect_anomaly(detect_anomaly)

    for epoch in range(1, epochs + 1):
        pinn.train()
        optimizer.zero_grad(set_to_none=True)

        # PDE on u_total = u_bs + U
        loss_physics = pde_dynamic_x(
            pinn=pinn,
            N_points=N_pde,
            sigma_mode=sigma_mode,
        )

        # Terminal condition on u_total at tau=0
        loss_terminal = spot_terminal_condition_x(
            pinn=pinn,
            N_points=N_boundary,
            sigma_mode=sigma_mode,
        )

        # Boundary conditions on u_total
        loss_boundary = spot_boundary_conditions_x(
            pinn=pinn,
            N_points=N_boundary,
            sigma_mode=sigma_mode,
            lambda_price_low=1.0,
            lambda_delta_low=0.0,
            lambda_price_high=1.0,
            lambda_delta_high=0.1,
        )

        loss = (
            loss_physics
            + lambda_terminal * loss_terminal
            + lambda_boundary * loss_boundary
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite total loss at epoch {epoch}: "
                f"physics={loss_physics.item()}, "
                f"terminal={loss_terminal.item()}, "
                f"boundary={loss_boundary.item()}"
            )

        loss.backward()

        bad_grad = False
        for name, param in pinn.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"[BAD GRAD] epoch={epoch} param={name}")
                bad_grad = True
                break

        if bad_grad:
            raise RuntimeError(f"Non-finite gradient detected at epoch {epoch}")

        grad_norm = None
        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(pinn.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        bad_param = False
        for name, param in pinn.named_parameters():
            if not torch.isfinite(param).all():
                print(f"[BAD PARAM] epoch={epoch} param={name}")
                bad_param = True
                break

        if bad_param:
            raise RuntimeError(f"Non-finite parameter detected after step at epoch {epoch}")

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

        if (current_loss < best_loss):
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
                        "call_put": getattr(pinn, "call_put", None),
                        "x_min": getattr(pinn, "x_min", None),
                        "x_max": getattr(pinn, "x_max", None),
                        "v_max": getattr(pinn, "v_max", None),
                        "T": getattr(pinn, "T", None),
                        "r_max": getattr(pinn, "r_max", None),
                        "kappa_max": getattr(pinn, "kappa_max", None),
                        "theta_max": getattr(pinn, "theta_max", None),
                        "sigma_max": getattr(pinn, "sigma_max", None),
                    },
                    best_model_path,
                )

        if epoch % print_every == 0 or epoch == 1:
            grad_norm_str = f"{float(grad_norm):.3e}" if grad_norm is not None else "None"
            print(
                f"Epoch {epoch:6d} | "
                f"Time: {elapsed:10.2f}s | "
                f"LR: {current_lr:.3e} | "
                f"Total: {current_loss:.6e} | "
                f"PDE_x: {loss_physics.item():.6e} | "
                f"Terminal_x: {loss_terminal.item():.6e} | "
                f"Boundary_x: {loss_boundary.item():.6e} | "
                f"GradNorm: {grad_norm_str} | "
                f"Best: {best_loss:.6e} @ {best_epoch}"
            )

    history["best_loss"] = best_loss
    history["best_epoch"] = best_epoch

    if best_state_dict is not None:
        pinn.load_state_dict(best_state_dict)

    return history
