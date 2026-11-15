#!/usr/bin/env python3
"""
2-factor Bergomi |PINN − MC| heatmap on the same (S, τ) grid and style
as the 1-factor smile heatmaps (``heatmap_baseline.png``).

1F reference grid (from ``run_smile_heatmaps``):
  S ∈ [70, 140] step 2.5,  τ ∈ [0.25, 1.5] with 21 points.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from support_tools.bergomi_test_suite import error_norms, save_heatmap
from support_tools.model_wrapper import load_model, predict_price
from support_tools.monte_carlo_pricing_tools import bergomi2_mc_grid


def pinn_price_grid_2f(pinn, S_grid, tau_grid, K: float, market: dict) -> np.ndarray:
    out = np.empty((len(S_grid), len(tau_grid)), dtype=np.float64)
    kw = {
        "X1": market["X1"],
        "X2": market["X2"],
        "r": market["r"],
        "xi0": market["xi0"],
        "omega1": market["omega1"],
        "omega2": market["omega2"],
        "kappa1": market["kappa1"],
        "kappa2": market["kappa2"],
        "rho1": market["rho1"],
        "rho2": market["rho2"],
        "rho12": market["rho12"],
        "sigma_mode": market["sigma_mode"],
        "stationary": market["stationary"],
    }
    for i, S in enumerate(S_grid):
        for j, tau in enumerate(tau_grid):
            p = predict_price(pinn, S=float(S), K=K, tau=float(tau), **kw)
            out[i, j] = float(p.detach().cpu().reshape(-1)[0])
    return out


def title_fragment(market: dict) -> str:
    return (
        f"baseline: ξ₀={market['xi0']:g}, "
        f"ω₁={market['omega1']:g}, ω₂={market['omega2']:g}, "
        f"κ₁={market['kappa1']:g}, κ₂={market['kappa2']:g}, "
        f"ρ₁={market['rho1']:g}, ρ₂={market['rho2']:g}, ρ₁₂={market['rho12']:g}, "
        f"L₁={market['X1']:g}, L₂={market['X2']:g}"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/bergomi2.pt")
    p.add_argument("--out-dir", default="results/bergomi2")
    p.add_argument("--n-mc-paths", type=int, default=40_000)
    p.add_argument("--n-mc-steps", type=int, default=80)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--also-slides",
        action="store_true",
        help="Also copy the heatmap into slides/Graphs/abs_err_heatmap.png",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pinn, meta = load_model(args.checkpoint, device=args.device)
    pinn.eval()

    # Same lattice as 1-factor smile heatmaps
    S_grid = np.arange(70.0, 140.0 + 0.5 * 2.5, 2.5, dtype=np.float64)
    tau_grid = np.linspace(0.25, 1.5, 21, dtype=np.float64)

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
        sigma_mode=meta.get("sigma_mode", "flat_fwd"),
        stationary=bool(meta.get("stationary", True)),
        K=100.0,
    )

    print(
        f"Grid: S[{S_grid.min():.1f},{S_grid.max():.1f}]×{S_grid.size}, "
        f"τ[{tau_grid.min():.2f},{tau_grid.max():.2f}]×{tau_grid.size} "
        f"({S_grid.size * tau_grid.size} pts)"
    )

    t0 = time.time()
    mc, se = bergomi2_mc_grid(
        S_grid=S_grid,
        tau_grid=tau_grid,
        K=market["K"],
        r=market["r"],
        q=0.0,
        xi0=market["xi0"],
        omega1=market["omega1"],
        omega2=market["omega2"],
        kappa1=market["kappa1"],
        kappa2=market["kappa2"],
        rho1=market["rho1"],
        rho2=market["rho2"],
        rho12=market["rho12"],
        X1_0=market["X1"],
        X2_0=market["X2"],
        n_paths=args.n_mc_paths,
        n_steps=args.n_mc_steps,
        stationary=market["stationary"],
        seed=0,
    )
    print(f"MC grid done in {time.time() - t0:.1f}s")

    t1 = time.time()
    pinn_p = pinn_price_grid_2f(pinn, S_grid, tau_grid, market["K"], market)
    print(f"PINN grid done in {time.time() - t1:.1f}s")

    abs_err = np.abs(pinn_p - mc)
    norms = error_norms(abs_err)

    heat_path = out_dir / "abs_err_heatmap.png"
    save_heatmap(
        S_grid,
        tau_grid,
        abs_err,
        title=(
            "Price-error norm |PINN − MC| on (τ, S)\n"
            f"Smile params — {title_fragment(market)}"
        ),
        path=heat_path,
        cbar_label="|ΔV| (price norm)",
        cmap="magma",
    )

    np.savez_compressed(
        out_dir / "fine_grid_abs_err.npz",
        S_grid=S_grid,
        tau_grid=tau_grid,
        pinn=pinn_p,
        mc=mc,
        mc_se=se,
        abs_err=abs_err,
    )

    report = {
        "checkpoint": args.checkpoint,
        "grid": {
            "S": [float(S_grid.min()), float(S_grid.max()), 2.5, int(S_grid.size)],
            "tau": [float(tau_grid.min()), float(tau_grid.max()), int(tau_grid.size)],
            "n_mc_paths": args.n_mc_paths,
            "n_mc_steps": args.n_mc_steps,
            "matched_1f_smile_heatmap_grid": True,
        },
        "market": market,
        "abs_norms": norms,
        "mean_mc_stderr": float(np.mean(se)),
        "max_mc_stderr": float(np.max(se)),
        "heatmap": str(heat_path),
    }
    with open(out_dir / "fine_grid_heatmap_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"RMSE={norms['RMSE']:.4f}  MAE={norms['MAE']:.4f}  "
        f"Linf={norms['Linf']:.4f}  → {heat_path}"
    )

    if args.also_slides:
        slides_path = _ROOT / "slides" / "Graphs" / "abs_err_heatmap.png"
        slides_path.parent.mkdir(parents=True, exist_ok=True)
        slides_path.write_bytes(heat_path.read_bytes())
        print(f"Copied to {slides_path}")


if __name__ == "__main__":
    main()
