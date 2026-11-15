#!/usr/bin/env python3
"""
Architecture + feature-set hillclimb for 2-factor Bergomi PINN.

Sweeps (feature_set, hidden, depth), scores against a fixed MC reference grid,
then fully trains the winner into ``checkpoints/bergomi2.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pricing.bergomi2_option_pricing import (
    FEATURE_SETS,
    PINN,
    _build_mc_anchor_batch,
    train_network,
)
from support_tools.monte_carlo_pricing_tools import bergomi2_mc_grid
from support_tools.model_wrapper import predict_price


EVAL_MARKET = dict(
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
    sigma_mode="flat_fwd",
    stationary=True,
    K=100.0,
)


def _make_pinn(hidden: int, depth: int, feature_set: str, device) -> PINN:
    return PINN(
        x_min=-3.0,
        x_max=3.0,
        X1_max=3.0,
        X2_max=3.0,
        T=2.0,
        r_max=0.15,
        xi0_max=0.12,
        omega_max=2.0,
        kappa_max=6.0,
        call_put="Call",
        hidden=hidden,
        depth=depth,
        v_max=1.0,
        kappa_floor=0.25,
        feature_set=feature_set,
    ).to(device)


def _pinn_grid(pinn, S_grid, tau_grid, market) -> np.ndarray:
    out = np.empty((len(S_grid), len(tau_grid)), dtype=np.float64)
    kw = {k: market[k] for k in (
        "X1", "X2", "r", "xi0", "omega1", "omega2", "kappa1", "kappa2",
        "rho1", "rho2", "rho12", "sigma_mode", "stationary",
    )}
    for i, S in enumerate(S_grid):
        for j, tau in enumerate(tau_grid):
            p = predict_price(
                pinn, S=float(S), K=market["K"], tau=float(tau), **kw
            )
            out[i, j] = float(p.detach().cpu().reshape(-1)[0])
    return out


def evaluate(pinn, S_grid, tau_grid, mc_ref, market) -> dict:
    pinn.eval()
    pinn_p = _pinn_grid(pinn, S_grid, tau_grid, market)
    err = np.abs(pinn_p - mc_ref)
    i_atm = int(np.argmin(np.abs(S_grid - market["K"])))
    j_t1 = int(np.argmin(np.abs(tau_grid - 1.0)))
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(err)),
        "Max_err": float(np.max(err)),
        "ATM_T1_PINN": float(pinn_p[i_atm, j_t1]),
        "ATM_T1_MC": float(mc_ref[i_atm, j_t1]),
    }


# Compact sweep: feature sets × promising widths/depths
DEFAULT_CONFIGS = [
    # v1 baseline arch
    {"name": "v1_h96_d4", "feature_set": "v1", "hidden": 96, "depth": 4},
    {"name": "v1_h128_d3", "feature_set": "v1", "hidden": 128, "depth": 3},
    {"name": "v1_h128_d4", "feature_set": "v1", "hidden": 128, "depth": 4},
    {"name": "v1_h160_d4", "feature_set": "v1", "hidden": 160, "depth": 4},
    {"name": "v1_h96_d5", "feature_set": "v1", "hidden": 96, "depth": 5},
    # curated extras
    {"name": "v1b_h96_d4", "feature_set": "v1b", "hidden": 96, "depth": 4},
    {"name": "v1b_h128_d3", "feature_set": "v1b", "hidden": 128, "depth": 3},
    {"name": "v1b_h128_d4", "feature_set": "v1b", "hidden": 128, "depth": 4},
    {"name": "v1b_h160_d4", "feature_set": "v1b", "hidden": 160, "depth": 4},
    # full engineered set (known weaker at short horizon — keep one probe)
    {"name": "v2_h128_d3", "feature_set": "v2", "hidden": 128, "depth": 3},
    {"name": "v2_h128_d4", "feature_set": "v2", "hidden": 128, "depth": 4},
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--full-epochs", type=int, default=1400)
    p.add_argument("--n-pde", type=int, default=3000)
    p.add_argument("--n-boundary", type=int, default=1000)
    p.add_argument("--mc-anchor-paths", type=int, default=20_000)
    p.add_argument("--mc-eval-paths", type=int, default=40_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="results/bergomi2/hillclimb")
    p.add_argument("--checkpoint", default="checkpoints/bergomi2.pt")
    p.add_argument("--skip-full-train", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    S_grid = np.arange(80.0, 130.0 + 0.5, 10.0, dtype=np.float64)
    tau_grid = np.linspace(0.25, 1.5, 6, dtype=np.float64)

    print("Feature sets:", {k: len(v) for k, v in FEATURE_SETS.items()})
    print("Building shared MC eval + anchor grids…", flush=True)
    t0 = time.time()
    mc_ref, _ = bergomi2_mc_grid(
        S_grid=S_grid,
        tau_grid=tau_grid,
        K=EVAL_MARKET["K"],
        r=EVAL_MARKET["r"],
        q=0.0,
        xi0=EVAL_MARKET["xi0"],
        omega1=EVAL_MARKET["omega1"],
        omega2=EVAL_MARKET["omega2"],
        kappa1=EVAL_MARKET["kappa1"],
        kappa2=EVAL_MARKET["kappa2"],
        rho1=EVAL_MARKET["rho1"],
        rho2=EVAL_MARKET["rho2"],
        rho12=EVAL_MARKET["rho12"],
        n_paths=args.mc_eval_paths,
        n_steps=80,
        stationary=True,
        seed=0,
    )
    anchor = _build_mc_anchor_batch(
        device, torch.float32, n_paths=args.mc_anchor_paths, n_steps=64, seed=7
    )
    print(f"MC grids ready in {time.time() - t0:.1f}s", flush=True)

    # Score archived baseline if present
    legacy = _ROOT / "checkpoints/legacy/bergomi2_h96_d4_f28.pt"
    baseline_row = None
    if legacy.exists():
        from support_tools.model_wrapper import load_model

        pinn_base, _ = load_model(str(legacy), device=device)
        baseline_row = {
            "name": "legacy_v1_h96_d4",
            "feature_set": "v1",
            "hidden": 96,
            "depth": 4,
            "n_features": 28,
            **evaluate(pinn_base, S_grid, tau_grid, mc_ref, EVAL_MARKET),
            "checkpoint": str(legacy),
        }
        print(
            f"Legacy baseline RMSE={baseline_row['RMSE']:.4f} "
            f"Max={baseline_row['Max_err']:.4f}",
            flush=True,
        )

    results = []
    for cfg in DEFAULT_CONFIGS:
        name = cfg["name"]
        feature_set, hidden, depth = cfg["feature_set"], cfg["hidden"], cfg["depth"]
        ckpt_path = out_dir / f"{name}.pt"
        print(
            f"\n=== {name}: set={feature_set}, hidden={hidden}, depth={depth} ===",
            flush=True,
        )
        pinn = _make_pinn(hidden, depth, feature_set, device)
        n_params = sum(x.numel() for x in pinn.parameters())
        t1 = time.time()
        hist = train_network(
            pinn,
            N_pde=args.n_pde,
            N_boundary=args.n_boundary,
            epochs=args.epochs,
            lr=1e-4,
            sigma_mode="flat_fwd",
            stationary=True,
            detach_source=False,
            relative_bs=True,
            use_mc_anchor=True,
            mc_anchor_batch=anchor,
            lambda_data=2.0,
            print_every=max(args.epochs // 5, 50),
            best_model_path=str(ckpt_path),
            save_model=True,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pinn.load_state_dict(ckpt["model_state_dict"])
        metrics = evaluate(pinn, S_grid, tau_grid, mc_ref, EVAL_MARKET)
        row = {
            "name": name,
            "feature_set": feature_set,
            "hidden": hidden,
            "depth": depth,
            "n_features": len(FEATURE_SETS[feature_set]),
            "n_params": n_params,
            "epochs": args.epochs,
            "best_epoch": hist["best_epoch"],
            "best_selection_loss": hist["best_loss"],
            "train_seconds": time.time() - t1,
            **metrics,
            "checkpoint": str(ckpt_path),
        }
        results.append(row)
        print(
            f"→ {name}: RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} "
            f"Max={metrics['Max_err']:.4f} ATM={metrics['ATM_T1_PINN']:.3f}"
            f" (MC {metrics['ATM_T1_MC']:.3f})",
            flush=True,
        )
        with open(out_dir / "hillclimb_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
                    "legacy_baseline": baseline_row,
                    "results": results,
                },
                f,
                indent=2,
            )

    ranked = sorted(results, key=lambda r: (r["RMSE"], r["Max_err"]))
    winner = ranked[0]
    print("\n=== Ranking (by RMSE) ===")
    if baseline_row is not None:
        print(
            f"legacy: RMSE={baseline_row['RMSE']:.4f} Max={baseline_row['Max_err']:.4f}"
        )
    for i, r in enumerate(ranked, 1):
        print(
            f"{i}. {r['name']}: RMSE={r['RMSE']:.4f} Max={r['Max_err']:.4f} "
            f"params={r['n_params']:,}"
        )
    print(f"Winner: {winner['name']}", flush=True)

    # Only promote if we beat legacy (when available)
    promote = True
    if baseline_row is not None and winner["RMSE"] > baseline_row["RMSE"] * 0.98:
        # allow tiny noise; require clear win otherwise keep training winner anyway
        # but warn
        if winner["RMSE"] >= baseline_row["RMSE"]:
            print(
                f"WARNING: winner RMSE {winner['RMSE']:.4f} does not beat "
                f"legacy {baseline_row['RMSE']:.4f}; still full-training winner.",
                flush=True,
            )

    summary = {
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "legacy_baseline": baseline_row,
        "results": results,
        "ranking": [r["name"] for r in ranked],
        "winner": winner,
    }

    if not args.skip_full_train:
        print(
            f"\n=== Full train winner {winner['name']} "
            f"for {args.full_epochs} epochs ===",
            flush=True,
        )
        pinn = _make_pinn(
            winner["hidden"], winner["depth"], winner["feature_set"], device
        )
        hist = train_network(
            pinn,
            N_pde=3500,
            N_boundary=1200,
            epochs=args.full_epochs,
            lr=1e-4,
            sigma_mode="flat_fwd",
            stationary=True,
            detach_source=False,
            relative_bs=True,
            use_mc_anchor=True,
            mc_anchor_batch=anchor,
            lambda_data=2.0,
            print_every=100,
            best_model_path=args.checkpoint,
            save_model=True,
        )
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        pinn.load_state_dict(ckpt["model_state_dict"])
        final_metrics = evaluate(pinn, S_grid, tau_grid, mc_ref, EVAL_MARKET)
        summary["full_train"] = {
            "checkpoint": args.checkpoint,
            "feature_set": winner["feature_set"],
            "hidden": winner["hidden"],
            "depth": winner["depth"],
            "epochs": args.full_epochs,
            "best_epoch": hist["best_epoch"],
            "best_selection_loss": hist["best_loss"],
            **final_metrics,
        }
        print(
            f"Full train RMSE={final_metrics['RMSE']:.4f} "
            f"MAE={final_metrics['MAE']:.4f} Max={final_metrics['Max_err']:.4f} "
            f"ATM={final_metrics['ATM_T1_PINN']:.3f} vs MC {final_metrics['ATM_T1_MC']:.3f}",
            flush=True,
        )
        if baseline_row is not None:
            print(
                f"vs legacy RMSE {baseline_row['RMSE']:.4f} → {final_metrics['RMSE']:.4f}",
                flush=True,
            )

    with open(out_dir / "hillclimb_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'hillclimb_results.json'}")


if __name__ == "__main__":
    main()
