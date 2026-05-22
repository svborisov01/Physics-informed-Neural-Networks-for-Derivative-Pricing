import torch
import torch.nn as nn
import numpy as np
import time
import copy
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Set random seeds for reproducibility
# ============================================================
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# PDE / conditions in log-moneyness x = log(S / K)
#
# We learn normalized price:
#     u(x, tau, r, sigma) = V(S, K, tau, r, sigma) / K
#
# For Black-Scholes/HD spot dynamics in x-space:
#     u_tau = 0.5 sigma^2 u_xx + (r - 0.5 sigma^2) u_x - r u
#
# so residual is
#     -u_tau + (r - 0.5 sigma^2) u_x + 0.5 sigma^2 u_xx - r u = 0
#
# ============================================================

def pde_dynamic_x(pinn, Npoints=5000):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    x = (pinn.x_min + (pinn.x_max - pinn.x_min) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    tau = (1e-4 + (pinn.T - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    r = (1e-4 + (pinn.r_max - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    sigma = (1e-4 + (pinn.sigma_max - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)

    def bs_operator(f):
        ftau = torch.autograd.grad(f.sum(), tau, create_graph=True)[0]
        fx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        fxx = torch.autograd.grad(fx.sum(), x, create_graph=True)[0]

        residual = ftau - 0.5 * sigma**2 * fxx - (r - 0.5 * sigma**2) * fx + r * f
        return residual

    u = pinn.forward_x(x, tau, r, sigma)
    residual = bs_operator(u)

    return torch.mean(residual**2)


def spot_terminal_condition_x(pinn, Npoints=2000):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    x = (pinn.x_min + (pinn.x_max - pinn.x_min) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    tau = torch.zeros(Npoints, 1, device=device, dtype=dtype)
    r = (1e-4 + (pinn.r_max - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    sigma = (1e-4 + (pinn.sigma_max - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)

    u = pinn.forward_x(x, tau, r, sigma)

    cp = pinn.call_put.lower()
    if cp == "call":
        payoff = torch.relu(torch.exp(x) - 1.0)
    elif cp == "put":
        payoff = torch.relu(1.0 - torch.exp(x))
    else:
        raise ValueError("pinn.call_put must be either 'Call' or 'Put'")

    return torch.mean((u - payoff)**2 / (1.0 + payoff)**2)


def spot_boundary_conditions_x(
    pinn,
    Npoints=2000,
    lambdapricelow=1.0,
    lambdadeltalow=0.0,
    lambdapricehigh=1.0,
    lambdadeltahigh=0.1,
):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    tau = (1e-4 + (pinn.T - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    r = (1e-4 + (pinn.r_max - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    sigma = (1e-4 + (pinn.sigma_max - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)

    xlow = torch.full((Npoints, 1), pinn.x_min, device=device, dtype=dtype).requires_grad_(True)
    xhigh = torch.full((Npoints, 1), pinn.x_max, device=device, dtype=dtype).requires_grad_(True)

    ulow = pinn.forward_x(xlow, tau, r, sigma)
    uxlow = torch.autograd.grad(ulow.sum(), xlow, create_graph=True)[0]

    uhigh = pinn.forward_x(xhigh, tau, r, sigma)
    uxhigh = torch.autograd.grad(uhigh.sum(), xhigh, create_graph=True)[0]

    cp = pinn.call_put.lower()

    if cp == "call":
        # x -> -inf : u -> 0, u_x -> 0
        targetpricelow = torch.zeros_like(ulow)
        targetdeltalow = torch.zeros_like(uxlow)

        # x -> +inf : u -> exp(x) - exp(-r tau), u_x -> exp(x)
        targetpricehigh = torch.exp(xhigh) - torch.exp(-r * tau)
        targetdeltahigh = torch.exp(xhigh)

    elif cp == "put":
        # x -> -inf : u -> exp(-r tau), u_x -> 0
        targetpricelow = torch.exp(-r * tau)
        targetdeltalow = torch.zeros_like(uxlow)

        # x -> +inf : u -> 0, u_x -> 0
        targetpricehigh = torch.zeros_like(uhigh)
        targetdeltahigh = torch.zeros_like(uxhigh)

    else:
        raise ValueError("pinn.call_put must be either 'Call' or 'Put'")

    losspricelow  = torch.mean((ulow  - targetpricelow )**2 / (1.0 + torch.abs(targetpricelow ))**2)
    lossdeltalow  = torch.mean((uxlow - targetdeltalow )**2 / (1.0 + torch.abs(targetdeltalow ))**2)
    losspricehigh = torch.mean((uhigh - targetpricehigh)**2 / (1.0 + torch.abs(targetpricehigh))**2)
    lossdeltahigh = torch.mean((uxhigh - targetdeltahigh)**2 / (1.0 + torch.abs(targetdeltahigh))**2)
    # print(lossdeltahigh.item(), lossdeltalow.item(), losspricehigh.item(), losspricelow.item())
    # print(uhigh.mean(), uxhigh.mean())

    return (
        lambdapricelow * losspricelow
        + lambdadeltalow * lossdeltalow
        + lambdapricehigh * losspricehigh
        + lambdadeltahigh * lossdeltahigh
    )


def american_constraint_x(pinn, Npoints=2000):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    x = (pinn.x_min + (pinn.x_max - pinn.x_min) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    tau = (1e-4 + (pinn.T - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    r = (1e-4 + (pinn.r_max - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)
    sigma = (1e-4 + (pinn.sigma+max - 1e-4) * torch.rand(Npoints, 1, device=device, dtype=dtype)).requires_grad_(True)

    u = pinn.forward_x(x, tau, r, sigma)

    cp = pinn.call_put.lower()
    if cp == "call":
        payoff = torch.relu(torch.exp(x) - 1.0)
    elif cp == "put":
        payoff = torch.relu(1.0 - torch.exp(x))
    else:
        raise ValueError("pinn.call_put must be either 'Call' or 'Put'")

    violation = payoff - u
    return torch.mean(torch.relu(violation)**2)


# ============================================================
# PINN model in the style of heston_option_pricing.py
# ============================================================

class PINN(nn.Module):
    """
    PINN for normalized HD/Black-Scholes option price

        u(x, tau, r, sigma) = V / K
        x = log(S / K)

    No analytical baseline is used.
    The architecture enforces the correct asymptotic shape directly.
    """

    def __init__(
        self,
        x_min,
        x_max,
        T,
        r_max,
        sigma_max,
        call_put="Call",
        hidden=128,
        depth=4,
    ):
        super().__init__()

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.x_scale = max(abs(x_min), abs(x_max), 1e-6)
        self.T = float(T)
        self.r_max = float(r_max)
        self.sigma_max = float(sigma_max)
        self.call_put = call_put
        self.depth = depth
        self.hidden = hidden

        input_dim = 12
        trunk = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            trunk += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.trunk = nn.Sequential(*trunk)

        self.head_gate = nn.Linear(hidden, 1)
        self.head_corr = nn.Linear(hidden, 1)

    def _cp_feature(self, x):
        cp = self.call_put.lower()
        if cp == "call":
            return torch.ones_like(x)
        elif cp == "put":
            return -torch.ones_like(x)
        else:
            raise ValueError("call_put must be either 'Call' or 'Put'")

    def _features_from_x(self, x, tau, r, sigma):
        x_norm = x / self.x_scale
        tau_norm = tau / self.T
        r_norm = r / self.r_max
        sigma_norm = sigma / self.sigma_max

        x_sq = x_norm ** 2
        x_cube = x_norm ** 3
        abs_x = torch.abs(x_norm)
        tau_sq = tau_norm ** 2
        sigma_sq = sigma_norm ** 2
        x_tau = x_norm * tau_norm
        x_sigma = x_norm * sigma_norm
        r_sigma = r_norm * sigma_norm
        cp_feat = self._cp_feature(x)

        return torch.cat(
            [
                x_norm,
                tau_norm,
                r_norm,
                sigma_norm,
                x_sq,
                x_cube,
                abs_x,
                tau_sq,
                sigma_sq,
                x_tau,
                x_sigma,
                cp_feat,
            ],
            dim=1,
        )

    def forward_x(self, x, tau, r, sigma):
        features = self._features_from_x(x, tau, r, sigma)
        h = self.trunk(features)

        gate_raw = self.head_gate(h)
        corr_raw = self.head_corr(h)

        cp = self.call_put.lower()

        if cp == "call":
            intrinsic = torch.exp(x) - torch.exp(-r * tau)

            # Smooth switching gate from OTM to ITM
            atm_shift = x / (1.0 + 0.25 * sigma * torch.sqrt(torch.clamp(tau, min=1e-8)))
            gate = torch.sigmoid(2.0 * atm_shift + gate_raw)

            # Small flexible correction on top of gate, but bounded
            corr = 0.10 * torch.tanh(corr_raw)

            # Enforce [0,1]-like multiplier
            multiplier = torch.clamp(gate + corr, min=0.0, max=1.0)

            u = intrinsic * multiplier
            u = torch.clamp(u, min=0.0)
            return u

        elif cp == "put":
            intrinsic = torch.exp(-r * tau) - torch.exp(x)

            atm_shift = -x / (1.0 + 0.25 * sigma * torch.sqrt(torch.clamp(tau, min=1e-8)))
            gate = torch.sigmoid(2.0 * atm_shift + gate_raw)
            corr = 0.10 * torch.tanh(corr_raw)
            multiplier = torch.clamp(gate + corr, min=0.0, max=1.0)

            u = intrinsic * multiplier
            u = torch.clamp(u, min=0.0)
            return u

        else:
            raise ValueError("call_put must be either 'Call' or 'Put'")

    def forward(self, S, K, tau, r, sigma):
        eps = 1e-8
        x = torch.log(torch.clamp(S, min=eps) / torch.clamp(K, min=eps))
        return self.forward_x(x, tau, r, sigma)

    def predict_normalized_price(self, S, K, tau, r, sigma):
        S_t = torch.as_tensor(S, dtype=torch.float32).reshape(-1, 1)
        K_t = torch.as_tensor(K, dtype=torch.float32).reshape(-1, 1)
        tau_t = torch.as_tensor(tau, dtype=torch.float32).reshape(-1, 1)
        r_t = torch.as_tensor(r, dtype=torch.float32).reshape(-1, 1)
        sigma_t = torch.as_tensor(sigma, dtype=torch.float32).reshape(-1, 1)

        device = next(self.parameters()).device
        S_t, K_t, tau_t, r_t, sigma_t = [z.to(device) for z in (S_t, K_t, tau_t, r_t, sigma_t)]

        with torch.no_grad():
            u_hat = self.forward(S_t, K_t, tau_t, r_t, sigma_t)

        return u_hat

    def predict_price(self, S, K, tau, r, sigma):
        K_t = torch.as_tensor(K, dtype=torch.float32).reshape(-1, 1)
        device = next(self.parameters()).device
        K_t = K_t.to(device)

        u_hat = self.predict_normalized_price(S, K, tau, r, sigma)
        return K_t * u_hat


# ============================================================
# Training loop in the same general style / print format
# ============================================================

def train_network(
    pinn,
    N_pde=5000,
    N_boundary=2000,
    epochs=5000,
    lr=1e-4,
    lambda_boundary=1.0,
    lambda_terminal=1.0,
    lambda_american=0.0,
    grad_clip=1.0,
    print_every=100,
    best_model_path="bestpinnxspace.pt",
    save_model=True,
    weight_decay=1e-2,
    cosine_eta_min=1e-5,
    detectanomaly=False,
    device=None,
):
    if device is None:
        device = next(pinn.parameters()).device
    pinn.to(device)

    optimizer = torch.optim.AdamW(pinn.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=cosine_eta_min,
    )

    history = {
        "epoch": [],
        "elapsed_time": [],
        "total": [],
        "physics": [],
        "terminal": [],
        "boundary": [],
        "american": [],
        "lr": [],
        "best_loss": None,
        "best_epoch": None,
        "model_depth": pinn.depth,
        "model_width": pinn.hidden,
    }

    bestloss = float("inf")
    bestepoch = None
    beststatedict = None
    starttime = time.time()

    torch.autograd.set_detect_anomaly(detectanomaly)

    for epoch in range(1, epochs + 1):
        pinn.train()
        optimizer.zero_grad(set_to_none=True)

        lossphysics = pde_dynamic_x(pinn=pinn, Npoints=N_pde)
        lossterminal = spot_terminal_condition_x(pinn=pinn, Npoints=N_boundary)
        lossboundary = spot_boundary_conditions_x(
            pinn=pinn,
            Npoints=N_boundary,
            lambdapricelow=1.0,
            lambdadeltalow=0.1,
            lambdapricehigh=1.0,
            lambdadeltahigh=0.5,
        )

        if lambda_american > 0.0:
            lossamerican = american_constraint_x(pinn=pinn, Npoints=N_boundary)
        else:
            lossamerican = torch.zeros(1, device=device, dtype=next(pinn.parameters()).dtype).squeeze()

        loss = (
            lossphysics
            + lambda_terminal * lossterminal
            + lambda_boundary * lossboundary
            + lambda_american * lossamerican
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite total loss at epoch {epoch}: "
                f"physics={lossphysics.item()}, "
                f"terminal={lossterminal.item()}, "
                f"boundary={lossboundary.item()}, "
                f"american={lossamerican.item()}"
            )

        loss.backward()

        badgrad = False
        for name, param in pinn.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"BAD GRAD epoch={epoch} param={name}")
                badgrad = True
                break
        if badgrad:
            raise RuntimeError(f"Non-finite gradient detected at epoch {epoch}")

        gradnorm = None
        if grad_clip is not None:
            gradnorm = torch.nn.utils.clip_grad_norm_(pinn.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        badparam = False
        for name, param in pinn.named_parameters():
            if not torch.isfinite(param).all():
                print(f"BAD PARAM epoch={epoch} param={name}")
                badparam = True
                break
        if badparam:
            raise RuntimeError(f"Non-finite parameter detected after step at epoch {epoch}")

        elapsed = time.time() - starttime
        currentlr = optimizer.param_groups[0]["lr"]
        currentloss = loss.item()

        history["epoch"].append(epoch)
        history["elapsed_time"].append(elapsed)
        history["total"].append(currentloss)
        history["physics"].append(lossphysics.item())
        history["terminal"].append(lossterminal.item())
        history["boundary"].append(lossboundary.item())
        history["american"].append(lossamerican.item())
        history["lr"].append(currentlr)

        if currentloss < bestloss:
            bestloss = currentloss
            bestepoch = epoch
            beststatedict = copy.deepcopy(pinn.state_dict())

            if save_model:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": beststatedict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": bestloss,
                        "call_put": getattr(pinn, "call_put", None),
                        "xmin": getattr(pinn, "xmin", None),
                        "xmax": getattr(pinn, "xmax", None),
                        "T": getattr(pinn, "T", None),
                        "rmax": getattr(pinn, "rmax", None),
                        "sigmamax": getattr(pinn, "sigmamax", None),
                    },
                    best_model_path,
                )

        if epoch % print_every == 0 or epoch == 1:
            gradnormstr = f"{float(gradnorm):.3e}" if gradnorm is not None else "None"
            print(
                f"Epoch {epoch:6d} | "
                f"Time: {elapsed:10.2f}s | "
                f"LR: {currentlr:.3e} | "
                f"Total: {currentloss:.6e} | "
                f"PDE_x: {lossphysics.item():.6e} | "
                f"Terminal_x: {lossterminal.item():.6e} | "
                f"Boundary_x: {lossboundary.item():.6e} | "
                f"GradNorm: {gradnormstr} | "
                f"Best: {bestloss:.6e} @ {bestepoch}"
            )

    history["best_loss"] = bestloss
    history["best_epoch"] = bestepoch

    if beststatedict is not None:
        pinn.load_state_dict(beststatedict)

    return history