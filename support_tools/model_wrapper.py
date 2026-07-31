"""
Universal wrapper for PINN option-pricing models.

Provides a single interface for loading checkpoints, running accuracy tests,
visualizing results, and computing Greeks across all supported model types:
  - two_d   : fixed-parameter Black–Scholes (2 inputs: x, tau)
  - hd      : variable r, sigma Black–Scholes (4 inputs: x, tau, r, sigma)
  - heston  : Heston correction model (price = u_bs + U)
  - bergomi : 1-factor Bergomi correction model (price = u_bs + U)
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from support_tools.analytical_pricing_tools import (
    bs_option_torch,
    heston_call_price,
    heston_put_price,
)


class ModelType(str, Enum):
    TWO_D = "two_d"
    HD = "hd"
    HESTON = "heston"
    BERGOMI = "bergomi"


# Known architecture overrides for bundled checkpoints (hidden, depth).
CHECKPOINT_ARCHITECTURES: Dict[str, Tuple[int, int]] = {
    "two_d.pt": (50, 5),
    "hd_minimal.pt": (100, 3),
    "hd_warm_restarts.pt": (40, 3),
    "long_training_heston.pt": (150, 6),
}

# Legacy checkpoints with incompatible input dimensions.
LEGACY_CHECKPOINTS = {"hd.pt", "heston.pt", "wide_moneyness_heston.pt"}


def detect_model_type(pinn: nn.Module) -> ModelType:
    """Detect model type from a PINN instance."""
    if hasattr(pinn, "X_max") and hasattr(pinn, "xi0_max"):
        return ModelType.BERGOMI
    if hasattr(pinn, "v_max") and hasattr(pinn, "kappa_max"):
        return ModelType.HESTON
    if hasattr(pinn, "r_max") and hasattr(pinn, "sigma_max"):
        return ModelType.HD
    if hasattr(pinn, "r") and hasattr(pinn, "sigma"):
        return ModelType.TWO_D
    raise ValueError(
        "Cannot detect model type. Expected a PINN from "
        "two_d_option_pricing, hd_option_pricing, heston_option_pricing, "
        "or bergomi_option_pricing."
    )


def detect_model_type_from_checkpoint(ckpt: dict) -> ModelType:
    """Detect model type from checkpoint metadata."""
    if "X_max" in ckpt and "xi0_max" in ckpt:
        return ModelType.BERGOMI
    if "v_max" in ckpt and "kappa_max" in ckpt:
        return ModelType.HESTON
    if "r_max" in ckpt and "sigma_max" in ckpt:
        return ModelType.HD
    if "r" in ckpt and "sigma" in ckpt:
        return ModelType.TWO_D
    raise ValueError("Cannot detect model type from checkpoint metadata.")


def infer_architecture(state_dict: dict, model_type: ModelType) -> Tuple[int, int]:
    """Infer (hidden, depth) from a state dict."""
    if model_type in (ModelType.TWO_D, ModelType.HD):
        hidden = state_dict["trunk.0.weight"].shape[0]
        layer_indices = [
            int(k.split(".")[1])
            for k in state_dict
            if k.startswith("trunk.") and k.endswith(".weight")
        ]
        depth = max(layer_indices) // 2 + 1
    else:
        hidden = state_dict["net.0.weight"].shape[0]
        layer_indices = [
            int(k.split(".")[1])
            for k in state_dict
            if k.startswith("net.") and k.endswith(".weight")
        ]
        depth = (max(layer_indices) + 1) // 2
    return hidden, depth


def _build_pinn(model_type: ModelType, ckpt: dict, hidden: int, depth: int) -> nn.Module:
    """Construct a PINN instance from checkpoint metadata."""
    call_put = ckpt.get("call_put", "Call")

    if model_type == ModelType.TWO_D:
        from pricing.two_d_option_pricing import PINN

        return PINN(
            x_min=ckpt["x_min"],
            x_max=ckpt["x_max"],
            T=ckpt["T"],
            r=ckpt["r"],
            sigma=ckpt["sigma"],
            call_put=call_put,
            hidden=hidden,
            depth=depth,
        )

    if model_type == ModelType.HD:
        from pricing.hd_option_pricing import PINN

        return PINN(
            x_min=ckpt["x_min"],
            x_max=ckpt["x_max"],
            T=ckpt["T"],
            r_max=ckpt["r_max"],
            sigma_max=ckpt["sigma_max"],
            call_put=call_put,
            hidden=hidden,
            depth=depth,
        )

    if model_type == ModelType.BERGOMI:
        from pricing.bergomi_option_pricing import PINN

        return PINN(
            x_min=ckpt["x_min"],
            x_max=ckpt["x_max"],
            X_max=ckpt["X_max"],
            T=ckpt["T"],
            r_max=ckpt["r_max"],
            xi0_max=ckpt["xi0_max"],
            omega_max=ckpt["omega_max"],
            kappa_max=ckpt["kappa_max"],
            call_put=call_put,
            hidden=hidden,
            depth=depth,
            v_max=ckpt.get("v_max", 1.0),
            kappa_floor=ckpt.get("kappa_floor", 0.25),
        )

    from pricing.heston_option_pricing import PINN

    return PINN(
        x_min=ckpt["x_min"],
        x_max=ckpt["x_max"],
        v_max=ckpt["v_max"],
        T=ckpt["T"],
        r_max=ckpt["r_max"],
        kappa_max=ckpt["kappa_max"],
        theta_max=ckpt["theta_max"],
        sigma_max=ckpt["sigma_max"],
        call_put=call_put,
        hidden=hidden,
        depth=depth,
    )


def load_model(
    checkpoint_path: str,
    device: Optional[Union[str, torch.device]] = None,
    hidden: Optional[int] = None,
    depth: Optional[int] = None,
) -> Tuple[nn.Module, dict]:
    """
    Load a trained PINN from a checkpoint file.

    Returns
    -------
    pinn : nn.Module
        Model in eval mode on the requested device.
    metadata : dict
        Checkpoint metadata (excluding state dicts).
    """
    basename = os.path.basename(checkpoint_path)
    if basename in LEGACY_CHECKPOINTS:
        raise ValueError(
            f"Checkpoint '{basename}' uses a legacy architecture that is "
            "incompatible with the current code. Use one of: "
            f"{sorted(CHECKPOINT_ARCHITECTURES.keys())}."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = detect_model_type_from_checkpoint(ckpt)

    if hidden is None or depth is None:
        if basename in CHECKPOINT_ARCHITECTURES:
            default_hidden, default_depth = CHECKPOINT_ARCHITECTURES[basename]
            hidden = hidden if hidden is not None else default_hidden
            depth = depth if depth is not None else default_depth
        else:
            inferred_hidden, inferred_depth = infer_architecture(
                ckpt["model_state_dict"], model_type
            )
            hidden = hidden if hidden is not None else inferred_hidden
            depth = depth if depth is not None else inferred_depth

    pinn = _build_pinn(model_type, ckpt, hidden, depth)
    pinn.load_state_dict(ckpt["model_state_dict"])
    pinn.to(device)
    pinn.eval()

    metadata = {
        k: v
        for k, v in ckpt.items()
        if k not in ("model_state_dict", "optimizer_state_dict")
    }
    metadata["model_type"] = model_type.value

    return pinn, metadata


def get_model_label(pinn: nn.Module) -> str:
    """Human-readable label for plots."""
    model_type = detect_model_type(pinn)
    labels = {
        ModelType.TWO_D: "Black–Scholes (2D)",
        ModelType.HD: "Black–Scholes (HD)",
        ModelType.HESTON: "Heston",
        ModelType.BERGOMI: "Bergomi (1-factor)",
    }
    return labels[model_type]


def predict_price(pinn: nn.Module, S, K, tau, **kwargs) -> torch.Tensor:
    """
    Unified price prediction for any supported PINN.

    Extra keyword arguments depend on model type:
      - two_d   : (none; r and sigma are fixed on the model)
      - hd      : r, sigma
      - heston  : v, r, kappa, theta, sigma, rho, sigma_mode (optional)
      - bergomi : X, r, xi0, omega, kappa, rho, sigma_mode, stationary (optional)
    """
    model_type = detect_model_type(pinn)

    if model_type == ModelType.TWO_D:
        return pinn.predict_price(S=S, K=K, tau=tau)

    if model_type == ModelType.HD:
        r = kwargs.get("r", 0.05)
        sigma = kwargs.get("sigma", 0.2)
        return pinn.predict_price(S=S, K=K, tau=tau, r=r, sigma=sigma)

    if model_type == ModelType.BERGOMI:
        return pinn.predict_price(
            S=S,
            K=K,
            X=kwargs["X"],
            tau=tau,
            r=kwargs["r"],
            xi0=kwargs["xi0"],
            omega=kwargs["omega"],
            kappa=kwargs["kappa"],
            rho=kwargs["rho"],
            sigma_mode=kwargs.get("sigma_mode", "spot_var"),
            stationary=kwargs.get("stationary", True),
        )

    sigma_mode = kwargs.get("sigma_mode", "hybrid")
    return pinn.predict_price(
        S=S,
        K=K,
        v=kwargs["v"],
        tau=tau,
        r=kwargs["r"],
        kappa=kwargs["kappa"],
        theta=kwargs["theta"],
        sigma=kwargs["sigma"],
        rho=kwargs["rho"],
        sigma_mode=sigma_mode,
    )


def _analytical_price(
    pinn: nn.Module,
    spot: float,
    K: float,
    tau: float,
    device: torch.device,
    dtype: torch.dtype,
    **params,
) -> float:
    """Ground-truth price for the given model type."""
    model_type = detect_model_type(pinn)
    call_put = getattr(pinn, "call_put", "Call")

    if model_type == ModelType.BERGOMI:
        from support_tools.monte_carlo_pricing_tools import Bergomi_Monte_Carlo

        return Bergomi_Monte_Carlo(
            S0=spot,
            K=K,
            T=tau,
            r=params["r"],
            q=params.get("q", 0.0),
            xi0=params["xi0"],
            omega=params["omega"],
            kappa=params["kappa"],
            rho=params["rho"],
            X0=params.get("X", params.get("X0", 0.0)),
            call_put=call_put,
            n_paths=int(params.get("n_mc_paths", 20_000)),
            n_steps=int(params.get("n_mc_steps", 100)),
            stationary=params.get("stationary", True),
            seed=params.get("mc_seed", None),
            return_stderr=False,
        )

    if model_type == ModelType.HESTON:
        if call_put.lower() == "call":
            return heston_call_price(
                spot,
                K=K,
                T=tau,
                r=params["r"],
                q=0.0,
                v0=params["v"],
                kappa=params["kappa"],
                theta=params["theta"],
                sigma=params["sigma"],
                rho=params["rho"],
            )[0]
        return heston_put_price(
            spot,
            K=K,
            T=tau,
            r=params["r"],
            q=0.0,
            v0=params["v"],
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            rho=params["rho"],
            verbose=False,
        )[0]

    r = params.get("r", getattr(pinn, "r", 0.05))
    sigma = params.get("sigma", getattr(pinn, "sigma", 0.2))

    S_t = torch.tensor([spot], device=device, dtype=dtype)
    K_t = torch.tensor([K], device=device, dtype=dtype)
    tau_t = torch.tensor([tau], device=device, dtype=dtype)
    r_t = torch.tensor([r], device=device, dtype=dtype)
    sigma_t = torch.tensor([sigma], device=device, dtype=dtype)

    price = bs_option_torch(
        S=S_t, K=K_t, tau=tau_t, r=r_t, sigma_bs=sigma_t, call_put=call_put
    )
    return float(price.detach().cpu().reshape(-1)[0])


def get_default_test_params(pinn: nn.Module, metadata: Optional[dict] = None) -> dict:
    """Default slice-test parameters derived from model metadata."""
    model_type = detect_model_type(pinn)
    metadata = metadata or {}

    if model_type == ModelType.TWO_D:
        return {
            "r": pinn.r,
            "sigma": pinn.sigma,
            "K": 100.0,
            "tau_min": 0.05,
            "tau_max": pinn.T,
            "n_tau": 50,
            "s_min": 20.0,
            "s_max": 500.0,
            "s_step": 2.0,
        }

    if model_type == ModelType.HD:
        return {
            "r": 0.05,
            "sigma": 0.2,
            "K": 100.0,
            "tau_min": 0.05,
            "tau_max": pinn.T,
            "n_tau": 50,
            "s_min": 20.0,
            "s_max": 500.0,
            "s_step": 2.0,
        }

    if model_type == ModelType.BERGOMI:
        return {
            "X": 0.0,
            "r": 0.05,
            "xi0": 0.04,
            "omega": 1.0,
            "kappa": 2.0,
            "rho": -0.5,
            "sigma_mode": metadata.get("sigma_mode", "spot_var"),
            "stationary": metadata.get("stationary", True),
            "K": 100.0,
            "tau_min": 0.1,
            "tau_max": min(pinn.T, 2.0),
            "n_tau": 30,
            "s_min": 50.0,
            "s_max": 200.0,
            "s_step": 5.0,
            "n_mc_paths": 20_000,
            "n_mc_steps": 100,
            "mc_seed": 42,
            "q": 0.0,
        }

    sigma_mode = metadata.get("sigma_mode", "mean_reverting")
    return {
        "v": 0.3,
        "r": 0.2,
        "kappa": 2.0,
        "theta": 0.3,
        "sigma": 0.2,
        "rho": 0.7,
        "sigma_mode": sigma_mode,
        "K": 100.0,
        "tau_min": 0.2,
        "tau_max": 4.0,
        "n_tau": 50,
        "s_min": 50.0,
        "s_max": 700.0,
        "s_step": 10.0,
    }


def run_slice_test(
    pinn: nn.Module,
    return_values: bool = False,
    print_metrics: bool = True,
    **kwargs,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run a 2D (S, tau) accuracy test against a ground-truth pricer.

    Automatically selects the correct benchmark and prediction API
    based on model type (analytical BS / Heston COS / Bergomi MC).
    Pass model-specific parameters via kwargs or rely on defaults from
    ``get_default_test_params``.

    Returns
    -------
    (spot_grid, tau_grid, mse_grid, pinn_prices, anal_prices) when
    return_values=True, else None.
    """
    defaults = get_default_test_params(pinn)
    params = {**defaults, **kwargs}

    K = params.pop("K")
    tau_min = params.pop("tau_min")
    tau_max = params.pop("tau_max")
    n_tau = params.pop("n_tau")
    s_min = params.pop("s_min")
    s_max = params.pop("s_max")
    s_step = params.pop("s_step")

    # MC-only kwargs must not be forwarded to predict_price
    mc_keys = ("n_mc_paths", "n_mc_steps", "mc_seed", "q", "X0")
    predict_params = {k: v for k, v in params.items() if k not in mc_keys}

    model_type = detect_model_type(pinn)
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype

    tau_grid = np.linspace(tau_min, tau_max, n_tau)
    spot_grid = np.arange(s_min, s_max + 1e-8, s_step)

    res, pinn_prices, anal_prices = [], [], []

    for spot in spot_grid:
        row_mse, row_pinn, row_anal = [], [], []
        for tau_val in tau_grid:
            price_pinn_t = predict_price(pinn, S=spot, K=K, tau=tau_val, **predict_params)
            price_pinn = float(price_pinn_t.detach().cpu().reshape(-1)[0])

            price_analytical = _analytical_price(
                pinn, spot, K, tau_val, device, dtype, **params
            )

            mse = (price_pinn - price_analytical) ** 2
            row_pinn.append(price_pinn)
            row_anal.append(price_analytical)
            row_mse.append(mse)

        pinn_prices.append(row_pinn)
        anal_prices.append(row_anal)
        res.append(row_mse)

    res = np.array(res)
    pinn_prices = np.array(pinn_prices)
    anal_prices = np.array(anal_prices)

    if print_metrics:
        print(f"Model: {get_model_label(pinn)}")
        print("MSE =", np.mean(res))
        print("RMSE =", np.sqrt(np.mean(res)))
        print("MAE =", np.mean(np.abs(pinn_prices - anal_prices)))

    if return_values:
        return spot_grid, tau_grid, res, pinn_prices, anal_prices
    return None


def visualize_slice_test(
    results: Tuple[np.ndarray, ...],
    pinn: Optional[nn.Module] = None,
    values: str = "diff",
    title: Optional[str] = None,
):
    """
    Plot a 3D surface from slice-test results.

    Parameters
    ----------
    results : tuple
        Output of ``run_slice_test(..., return_values=True)``.
    pinn : nn.Module, optional
        Used to derive plot title when title is not provided.
    values : {"diff", "pinn", "anal"}
        Which surface to display.
    """
    from support_tools.graphing_tools import plot_3d_result

    if title is None and pinn is not None:
        title = f"Interactive {get_model_label(pinn)} PDE solution"
    plot_3d_result(results, values=values, title=title)


def compute_greeks(
    pinn: nn.Module,
    S,
    K,
    tau,
    device: Optional[torch.device] = None,
    create_graph: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute price, Delta (∂V/∂S), and Theta (∂V/∂t) via autograd.

    Extra kwargs by model type:
      - two_d   : none
      - hd      : r, sigma
      - heston  : v, r, kappa, theta, sigma, rho, sigma_mode
      - bergomi : X, r, xi0, omega, kappa, rho, sigma_mode, stationary

    Returns
    -------
    price, delta, theta : torch.Tensor, each shape (N, 1)
    """
    model_type = detect_model_type(pinn)

    if device is None:
        device = next(pinn.parameters()).device

    S_t = torch.as_tensor(S, dtype=torch.float32, device=device).reshape(-1, 1)
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device).reshape(-1, 1)
    tau_t = torch.as_tensor(tau, dtype=torch.float32, device=device).reshape(-1, 1)
    S_t.requires_grad_(True)
    tau_t.requires_grad_(True)

    if model_type == ModelType.TWO_D:
        u = pinn.forward(S_t, K_t, tau_t)
    elif model_type == ModelType.HD:
        r = kwargs.get("r", 0.05)
        sigma = kwargs.get("sigma", 0.2)
        r_t = torch.full_like(S_t, r)
        sigma_t = torch.full_like(S_t, sigma)
        u = pinn.forward(S_t, K_t, tau_t, r_t, sigma_t)
    elif model_type == ModelType.HESTON:
        from support_tools.analytical_pricing_tools import (
            bs_option_normalized_from_x,
            sigma_bs_effective,
        )

        v = kwargs["v"]
        r = kwargs["r"]
        kappa = kwargs["kappa"]
        theta = kwargs["theta"]
        sigma = kwargs["sigma"]
        rho = kwargs["rho"]
        sigma_mode = kwargs.get("sigma_mode", "hybrid")

        v_t = torch.full_like(S_t, v)
        r_t = torch.full_like(S_t, r)
        kappa_t = torch.full_like(S_t, kappa)
        theta_t = torch.full_like(S_t, theta)
        sigma_t = torch.full_like(S_t, sigma)
        rho_t = torch.full_like(S_t, rho)

        eps = 1e-8
        x_t = torch.log(torch.clamp(S_t, min=eps) / torch.clamp(K_t, min=eps))
        U = pinn.forward_x(x_t, v_t, tau_t, r_t, kappa_t, theta_t, sigma_t, rho_t)
        sigma_bs = sigma_bs_effective(
            v=v_t, theta=theta_t, kappa=kappa_t, tau=tau_t, mode=sigma_mode
        )
        u_bs = bs_option_normalized_from_x(
            x=x_t, tau=tau_t, r=r_t, sigma_bs=sigma_bs, call_put=pinn.call_put
        )
        u = u_bs + U
    elif model_type == ModelType.BERGOMI:
        from support_tools.analytical_pricing_tools import bs_option_normalized_from_x
        from pricing.bergomi_option_pricing import v_bergomi, sigma_bs_bergomi

        X = kwargs["X"]
        r = kwargs["r"]
        xi0 = kwargs["xi0"]
        omega = kwargs["omega"]
        kappa = kwargs["kappa"]
        rho = kwargs["rho"]
        sigma_mode = kwargs.get("sigma_mode", "spot_var")
        stationary = kwargs.get("stationary", True)

        X_t = torch.full_like(S_t, X)
        r_t = torch.full_like(S_t, r)
        xi0_t = torch.full_like(S_t, xi0)
        omega_t = torch.full_like(S_t, omega)
        kappa_t = torch.full_like(S_t, kappa)
        rho_t = torch.full_like(S_t, rho)

        eps = 1e-8
        x_t = torch.log(torch.clamp(S_t, min=eps) / torch.clamp(K_t, min=eps))
        U = pinn.forward_x(
            x_t, X_t, tau_t, r_t, xi0_t, omega_t, kappa_t, rho_t
        )

        t = None if stationary else (pinn.T - tau_t).clamp_min(0.0)
        v = v_bergomi(
            X_t,
            xi0_t,
            omega_t,
            kappa_t,
            t=t,
            stationary=stationary,
            v_max=getattr(pinn, "v_max", None),
        )
        sigma_bs = sigma_bs_bergomi(
            xi0=xi0_t, v=v, tau=tau_t, mode=sigma_mode
        )
        u_bs = bs_option_normalized_from_x(
            x=x_t, tau=tau_t, r=r_t, sigma_bs=sigma_bs, call_put=pinn.call_put
        )
        u = u_bs + U
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    V = u * K_t

    grad_S = torch.autograd.grad(
        V.sum(), S_t, create_graph=create_graph, retain_graph=True
    )[0]
    grad_tau = torch.autograd.grad(
        V.sum(), tau_t, create_graph=create_graph, retain_graph=True
    )[0]

    delta = grad_S
    theta = -grad_tau  # ∂V/∂t = -∂V/∂tau (tau = T - t)

    return V, delta, theta


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return norm.cdf(x)


def bs_greeks_analytic(
    S,
    K,
    tau,
    r: float,
    sigma: float,
    call_put: str = "call",
) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic Black–Scholes Delta and Theta."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)

    cp = call_put.lower()
    eps = 1e-12
    tau_eff = np.maximum(tau, eps)
    S_eff = np.maximum(S, eps)
    sigma_eff = max(sigma, eps)

    d1 = (np.log(S_eff / K) + (r + 0.5 * sigma_eff**2) * tau_eff) / (
        sigma_eff * np.sqrt(tau_eff)
    )
    d2 = d1 - sigma_eff * np.sqrt(tau_eff)

    if cp == "call":
        delta = _norm_cdf(d1)
        theta = (
            -(S_eff * sigma_eff * np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi * tau_eff))
            - r * K * np.exp(-r * tau_eff) * _norm_cdf(d2)
        )
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (
            -(S_eff * sigma_eff * np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi * tau_eff))
            + r * K * np.exp(-r * tau_eff) * _norm_cdf(-d2)
        )

    return delta, theta


def compare_greeks(
    pinn: nn.Module,
    K: float = 1.0,
    r: Optional[float] = None,
    sigma: Optional[float] = None,
    call_put: Optional[str] = None,
    m_min: float = 0.5,
    m_max: float = 1.5,
    n_m: int = 25,
    tau_min: float = 0.05,
    tau_max: Optional[float] = None,
    n_tau: int = 25,
    device: Optional[torch.device] = None,
    **kwargs,
):
    """
    Compare PINN Greeks against analytic Black–Scholes on an (S/K, tau) grid.

    Supported for two_d and hd models. For hd, pass r and sigma via kwargs
    or rely on defaults.
    """
    model_type = detect_model_type(pinn)
    if model_type == ModelType.HESTON:
        raise NotImplementedError(
            "Greek comparison against analytic BS is not defined for Heston models."
        )

    if device is None:
        device = next(pinn.parameters()).device

    if r is None:
        r = pinn.r if model_type == ModelType.TWO_D else kwargs.get("r", 0.05)
    if sigma is None:
        sigma = pinn.sigma if model_type == ModelType.TWO_D else kwargs.get("sigma", 0.2)
    if call_put is None:
        call_put = pinn.call_put
    if tau_max is None:
        tau_max = pinn.T

    m_grid = np.linspace(m_min, m_max, n_m)
    tau_grid = np.linspace(tau_min, tau_max, n_tau)
    M_grid, Tau_grid = np.meshgrid(m_grid, tau_grid)

    S_grid = M_grid * K
    S_flat = S_grid.ravel()
    K_flat = np.full_like(S_flat, K)
    Tau_flat = Tau_grid.ravel()

    greek_kwargs = {"r": r, "sigma": sigma} if model_type == ModelType.HD else {}
    _, delta_pinn_flat, theta_pinn_flat = compute_greeks(
        pinn,
        S=S_flat,
        K=K_flat,
        tau=Tau_flat,
        device=device,
        create_graph=False,
        **greek_kwargs,
    )

    delta_pinn = delta_pinn_flat.cpu().numpy().reshape(n_tau, n_m)
    theta_pinn = theta_pinn_flat.cpu().numpy().reshape(n_tau, n_m)

    delta_anal, theta_anal = bs_greeks_analytic(
        S=S_grid, K=K, tau=Tau_grid, r=r, sigma=sigma, call_put=call_put
    )

    delta_diff = delta_pinn - delta_anal
    theta_diff = theta_pinn - theta_anal

    return {
        "delta_anal": torch.as_tensor(delta_anal, dtype=torch.float32),
        "delta_pinn": torch.as_tensor(delta_pinn, dtype=torch.float32),
        "delta_diff": torch.as_tensor(delta_diff, dtype=torch.float32),
        "theta_anal": torch.as_tensor(theta_anal, dtype=torch.float32),
        "theta_pinn": torch.as_tensor(theta_pinn, dtype=torch.float32),
        "theta_diff": torch.as_tensor(theta_diff, dtype=torch.float32),
        "M_grid": M_grid,
        "Tau_grid": Tau_grid,
    }
