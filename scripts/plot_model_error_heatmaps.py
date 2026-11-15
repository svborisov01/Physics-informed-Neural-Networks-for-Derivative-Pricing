#!/usr/bin/env python3
"""
Abs-error heatmaps + metrics table for Heston, Bergomi 1F, Bergomi 2F,
and Bergomi 2F trained with QMC sample anchors — all on the same (S, τ) grid.

Grid (matches 1F smile heatmaps):
  S ∈ [70, 140] step 2.5,  τ ∈ [0.25, 1.5] × 21.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from support_tools.bergomi_test_suite import SmileParams, error_norms, pinn_price_grid, save_heatmap
from support_tools.model_wrapper import (
    _analytical_price,
    get_default_test_params,
    load_model,
    predict_price,
)
from support_tools.monte_carlo_pricing_tools import (
    bergomi_qmc_add_grid,
    bergomi2_qmc_add_grid,
)


def pinn_price_grid_2f(pinn, S_grid, tau_grid, K: float, market: dict) -> np.ndarray:
    out = np.empty((len(S_grid), len(tau_grid)), dtype=np.float64)
    kw = {k: market[k] for k in (
        "X1", "X2", "r", "xi0", "omega1", "omega2", "kappa1", "kappa2",
        "rho1", "rho2", "rho12", "sigma_mode", "stationary",
    )}
    for i, S in enumerate(S_grid):
        for j, tau in enumerate(tau_grid):
            p = predict_price(pinn, S=float(S), K=K, tau=float(tau), **kw)
            out[i, j] = float(p.detach().cpu().reshape(-1)[0])
    return out


def heston_grids(pinn, S_grid, tau_grid, K: float, params: dict):
    device = next(pinn.parameters()).device
    dtype = next(pinn.parameters()).dtype
    pinn_p = np.empty((len(S_grid), len(tau_grid)), dtype=np.float64)
    ref = np.empty_like(pinn_p)
    pred_kw = {
        "r": params["r"],
        "v": params["v"],
        "kappa": params["kappa"],
        "theta": params["theta"],
        "sigma": params["sigma"],
        "rho": params["rho"],
        "sigma_mode": params.get("sigma_mode", "hybrid"),
    }
    for i, S in enumerate(S_grid):
        for j, tau in enumerate(tau_grid):
            pinn_p[i, j] = float(
                predict_price(pinn, S=float(S), K=K, tau=float(tau), **pred_kw)
                .detach()
                .cpu()
            )
            ref[i, j] = _analytical_price(
                pinn, float(S), K, float(tau), device, dtype, **pred_kw
            )
    return pinn_p, ref


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heston-ckpt", default="checkpoints/long_training_heston.pt")
    p.add_argument("--bergomi1-ckpt", default="checkpoints/bergomi.pt")
    p.add_argument("--bergomi2-ckpt", default="checkpoints/bergomi2.pt")
    p.add_argument(
        "--bergomi2-qmc-aid-ckpt",
        default="checkpoints/bergomi2_qmc_aid.pt",
    )
    p.add_argument("--out-dir", default="results/fine_grid_comparison")
    p.add_argument("--n-qmc-paths", type=int, default=65_536)
    p.add_argument("--n-qmc-steps", type=int, default=128)
    p.add_argument("--also-slides", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slides = _ROOT / "slides" / "Graphs"

    S_grid = np.arange(70.0, 140.0 + 0.5 * 2.5, 2.5, dtype=np.float64)
    tau_grid = np.linspace(0.25, 1.5, 21, dtype=np.float64)
    K = 100.0

    rows = []

    # ---- Heston vs COS ----
    heston, _ = load_model(args.heston_ckpt, device=args.device)
    hp = get_default_test_params(heston)
    t0 = time.time()
    pinn_h, ref_h = heston_grids(heston, S_grid, tau_grid, K, hp)
    err_h = np.abs(pinn_h - ref_h)
    norms_h = error_norms(err_h)
    heat_h = out_dir / "heatmap_heston.png"
    save_heatmap(
        S_grid,
        tau_grid,
        err_h,
        title="Price-error norm |PINN − COS| on (τ, S)\nHeston baseline market",
        path=heat_h,
        cbar_label="|ΔV| (price norm)",
    )
    rows.append(
        {
            "model": "Heston",
            "benchmark": "COS",
            "training_aid": "—",
            "checkpoint": args.heston_ckpt,
            "abs_norms": norms_h,
            "heatmap": str(heat_h),
            "elapsed_s": time.time() - t0,
        }
    )
    print(f"Heston RMSE={norms_h['RMSE']:.4f} Max={norms_h['Linf']:.4f}")

    # ---- 1F vs QMC+ADD ----
    berg1, _ = load_model(args.bergomi1_ckpt, device=args.device)
    smile = SmileParams("baseline", xi0=0.04, omega=1.0, kappa=2.0, rho=-0.5)
    t0 = time.time()
    ref_1, se_1 = bergomi_qmc_add_grid(
        S_grid=S_grid,
        tau_grid=tau_grid,
        K=K,
        r=0.05,
        q=0.0,
        xi0=smile.xi0,
        omega=smile.omega,
        kappa=smile.kappa,
        rho=smile.rho,
        X0=0.0,
        n_paths=args.n_qmc_paths,
        n_steps=args.n_qmc_steps,
        stationary=True,
        seed=7,
    )
    pinn_1 = pinn_price_grid(berg1, S_grid, tau_grid, K, smile)
    err_1 = np.abs(pinn_1 - ref_1)
    norms_1 = error_norms(err_1)
    heat_1 = out_dir / "heatmap_bergomi1.png"
    save_heatmap(
        S_grid,
        tau_grid,
        err_1,
        title=(
            "Price-error norm |PINN − QMC+ADD| on (τ, S)\n"
            f"Smile params — {smile.title_fragment()}"
        ),
        path=heat_1,
        cbar_label="|ΔV| (price norm)",
    )
    rows.append(
        {
            "model": "Bergomi 1F",
            "benchmark": "QMC+ADD",
            "training_aid": "—",
            "checkpoint": args.bergomi1_ckpt,
            "abs_norms": norms_1,
            "mean_ref_stderr": float(np.mean(se_1)),
            "heatmap": str(heat_1),
            "elapsed_s": time.time() - t0,
        }
    )
    print(f"1F RMSE={norms_1['RMSE']:.4f} Max={norms_1['Linf']:.4f}")

    market_2 = dict(
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
        stationary=True,
    )
    t_ref = time.time()
    ref_2, se_2 = bergomi2_qmc_add_grid(
        S_grid=S_grid,
        tau_grid=tau_grid,
        K=K,
        q=0.0,
        n_paths=args.n_qmc_paths,
        n_steps=args.n_qmc_steps,
        seed=0,
        **{k: v for k, v in market_2.items() if k not in ("X1", "X2")},
        X1_0=market_2["X1"],
        X2_0=market_2["X2"],
    )
    print(f"2F QMC reference done in {time.time() - t_ref:.1f}s "
          f"(mean SE={float(np.mean(se_2)):.4f})")

    def eval_2f(ckpt: str, label: str, aid: str, out_name: str):
        if not Path(ckpt).exists():
            print(f"SKIP {label}: missing {ckpt}")
            return None
        pinn, meta = load_model(ckpt, device=args.device)
        mkt = {
            **market_2,
            "sigma_mode": meta.get("sigma_mode", "flat_fwd"),
            "stationary": bool(meta.get("stationary", True)),
        }
        t0 = time.time()
        pinn_p = pinn_price_grid_2f(pinn, S_grid, tau_grid, K, mkt)
        err = np.abs(pinn_p - ref_2)
        norms = error_norms(err)
        heat = out_dir / out_name
        save_heatmap(
            S_grid,
            tau_grid,
            err,
            title=(
                f"Price-error norm |PINN − QMC+ADD| on (τ, S)\n"
                f"{label} — ξ₀={mkt['xi0']:g}, ω₁={mkt['omega1']:g}, "
                f"ω₂={mkt['omega2']:g}, κ₁={mkt['kappa1']:g}, κ₂={mkt['kappa2']:g}"
            ),
            path=heat,
            cbar_label="|ΔV| (price norm)",
        )
        row = {
            "model": label,
            "benchmark": "QMC+ADD",
            "training_aid": aid,
            "checkpoint": ckpt,
            "abs_norms": norms,
            "mean_ref_stderr": float(np.mean(se_2)),
            "heatmap": str(heat),
            "elapsed_s": time.time() - t0,
            "meta_anchor": meta.get("mc_anchor_method"),
            "use_mc_anchor": meta.get("use_mc_anchor"),
        }
        print(f"{label} RMSE={norms['RMSE']:.4f} Max={norms['Linf']:.4f}")
        return row

    r2 = eval_2f(
        args.bergomi2_ckpt,
        "Bergomi 2F",
        "MC samples" if True else "—",
        "heatmap_bergomi2.png",
    )
    if r2 is not None:
        # Prefer checkpoint metadata for the aid label
        aid = r2.get("meta_anchor")
        if aid in {"qmc_add", "qmc", "add"}:
            r2["training_aid"] = "QMC+ADD samples"
        elif r2.get("use_mc_anchor"):
            r2["training_aid"] = "MC samples"
        else:
            r2["training_aid"] = "—"
        rows.append(r2)

    r2q = eval_2f(
        args.bergomi2_qmc_aid_ckpt,
        "Bergomi 2F + QMC aid",
        "QMC+ADD samples",
        "heatmap_bergomi2_qmc_aid.png",
    )
    if r2q is not None:
        rows.append(r2q)

    # Shared color scale optional: leave per-plot auto scale (matches existing style)

    table = []
    for r in rows:
        n = r["abs_norms"]
        table.append(
            {
                "model": r["model"],
                "benchmark": r["benchmark"],
                "training_aid": r["training_aid"],
                "RMSE_V": n["RMSE"],
                "Max_V": n["Linf"],
                "MAE_V": n["MAE"],
                "RMSE_u": n["RMSE"] / K,
                "Max_u": n["Linf"] / K,
                "heatmap": r["heatmap"],
            }
        )

    report = {
        "grid": {
            "S": [float(S_grid.min()), float(S_grid.max()), 2.5, int(S_grid.size)],
            "tau": [float(tau_grid.min()), float(tau_grid.max()), int(tau_grid.size)],
            "K": K,
            "n_qmc_paths": args.n_qmc_paths,
            "n_qmc_steps": args.n_qmc_steps,
        },
        "rows": rows,
        "table": table,
    }
    with open(out_dir / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Fine-grid comparison (same (S, tau) lattice)",
        "",
        f"Grid: S in [{S_grid.min():.0f}, {S_grid.max():.0f}] step 2.5, "
        f"tau in [{tau_grid.min():.2f}, {tau_grid.max():.2f}] x {tau_grid.size}, K={K:.0f}.",
        "",
        "| Model | Benchmark | Training aid | RMSE (u) | Max (u) | RMSE (V) | Max (V) |",
        "|-------|-----------|--------------|----------|---------|----------|---------|",
    ]
    for t in table:
        md_lines.append(
            f"| {t['model']} | {t['benchmark']} | {t['training_aid']} | "
            f"{t['RMSE_u']:.4f} | {t['Max_u']:.4f} | "
            f"{t['RMSE_V']:.4f} | {t['Max_V']:.4f} |"
        )
    md_lines.append("")
    (out_dir / "COMPARISON.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md_lines))

    if args.also_slides:
        slides.mkdir(parents=True, exist_ok=True)
        mapping = {
            "heatmap_heston.png": "heatmap_heston.png",
            "heatmap_bergomi1.png": "heatmap_baseline.png",
            "heatmap_bergomi2.png": "abs_err_heatmap.png",
            "heatmap_bergomi2_qmc_aid.png": "abs_err_heatmap_qmc_aid.png",
        }
        for src_name, dst_name in mapping.items():
            src = out_dir / src_name
            if src.exists():
                shutil.copy2(src, slides / dst_name)
                print(f"Copied {src.name} → slides/Graphs/{dst_name}")


if __name__ == "__main__":
    main()
