#!/usr/bin/env python3
"""Run MC accuracy tests for the trained 2-factor Bergomi PINN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from support_tools.model_wrapper import load_model, predict_price, run_slice_test
from support_tools.monte_carlo_pricing_tools import Bergomi2_Monte_Carlo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="trained_models/bergomi2.pt")
    p.add_argument("--out-dir", default="bergomi2")
    p.add_argument("--n-mc-paths", type=int, default=40_000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    pinn, meta = load_model(args.checkpoint, device=args.device)
    pinn.eval()
    print("Loaded", args.checkpoint, "| type=", meta.get("model_type"))

    # Default market (fast / slow factors)
    market = dict(
        X1=0.0,
        X2=0.0,
        r=0.05,
        xi0=0.04,
        omega1=1.0,
        omega2=0.6,
        kappa1=4.0,
        kappa2=0.8,
        rho1=-0.6,
        rho2=-0.3,
        rho12=0.2,
        sigma_mode=meta.get("sigma_mode", "spot_var"),
        stationary=meta.get("stationary", True),
        K=100.0,
        n_mc_paths=args.n_mc_paths,
        n_mc_steps=80,
        mc_seed=0,
        mc_method="mc",
        tau_min=0.25,
        tau_max=1.5,
        n_tau=6,
        s_min=80.0,
        s_max=130.0,
        s_step=10.0,
    )

    t0 = time.time()
    results = run_slice_test(pinn, return_values=True, print_metrics=True, **market)
    elapsed = time.time() - t0
    spot, tau, mse_grid, pinn_p, mc_p = results
    abs_e = np.abs(pinn_p - mc_p)
    report = {
        "checkpoint": args.checkpoint,
        "market": {k: v for k, v in market.items() if k != "mc_method"},
        "n_points": int(abs_e.size),
        "RMSE": float(np.sqrt(np.mean(abs_e ** 2))),
        "MAE": float(np.mean(abs_e)),
        "Max_err": float(np.max(abs_e)),
        "L1": float(np.mean(abs_e)),
        "Linf": float(np.max(abs_e)),
        "test_seconds": elapsed,
        "meta": {
            "hidden": meta.get("hidden"),
            "depth": meta.get("depth"),
            "best_epoch": meta.get("epoch"),
            "train_loss": meta.get("loss"),
            "n_features": meta.get("feature_names") and len(meta["feature_names"]),
            "feature_names": meta.get("feature_names"),
            "price_softplus_beta": meta.get("price_softplus_beta"),
        },
    }
    with open(art / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Error heatmap
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    im = ax.pcolormesh(tau, spot, abs_e, shading="auto", cmap="magma")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$S$")
    ax.set_title(
        f"2F Bergomi |PINN − MC|  RMSE={report['RMSE']:.3f}, Max={report['Max_err']:.3f}"
    )
    fig.colorbar(im, ax=ax, label="|ΔV|")
    fig.tight_layout()
    fig.savefig(art / "abs_err_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # ATM term structure vs MC
    K = market["K"]
    i_atm = int(np.argmin(np.abs(spot - K)))
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    ax.plot(tau, pinn_p[i_atm], "o-", label="PINN")
    ax.plot(tau, mc_p[i_atm], "s--", label="MC")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Call price")
    ax.set_title(f"ATM term structure (S=K={K:.0f})")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(art / "atm_term_structure.png", bbox_inches="tight")
    plt.close(fig)

    # Single-point smoke check
    price = float(
        predict_price(
            pinn,
            S=100,
            K=100,
            tau=1.0,
            X1=0,
            X2=0,
            r=0.05,
            xi0=0.04,
            omega1=1.0,
            omega2=0.6,
            kappa1=4.0,
            kappa2=0.8,
            rho1=-0.6,
            rho2=-0.3,
            rho12=0.2,
        )
        .detach()
        .cpu()
    )
    mc = Bergomi2_Monte_Carlo(
        S0=100,
        K=100,
        T=1.0,
        r=0.05,
        q=0.0,
        xi0=0.04,
        omega1=1.0,
        omega2=0.6,
        kappa1=4.0,
        kappa2=0.8,
        rho1=-0.6,
        rho2=-0.3,
        rho12=0.2,
        n_paths=80_000,
        n_steps=100,
        seed=1,
        return_stderr=True,
    )
    print(f"ATM T=1: PINN={price:.4f}  MC={mc[0]:.4f} ± {mc[1]:.4f}")
    report["atm_T1"] = {"pinn": price, "mc": mc[0], "mc_se": mc[1]}
    with open(art / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: report[k] for k in ("RMSE", "MAE", "Max_err", "test_seconds")}, indent=2))


if __name__ == "__main__":
    main()
