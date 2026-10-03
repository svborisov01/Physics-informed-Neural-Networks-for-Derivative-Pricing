"""
Extensive Bergomi PINN validation against QMC + antithetic differencing (ADD).

Produces:
  1. Fine-grid absolute / relative price-error norms (L1, L2, Linf)
  2. Smile-structure heatmaps in (tau, S) for several (omega, rho, kappa, xi0)
  3. PINN vs QMC+ADD Greeks (Delta, Theta) tables and heatmaps
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from support_tools.model_wrapper import compute_greeks, load_model, predict_price
from support_tools.monte_carlo_pricing_tools import (
    Bergomi_QMC_ADD_Greeks,
    bergomi_qmc_add_grid,
)


@dataclass(frozen=True)
class SmileParams:
    name: str
    xi0: float
    omega: float
    kappa: float
    rho: float
    X: float = 0.0
    r: float = 0.05
    q: float = 0.0
    stationary: bool = True
    sigma_mode: str = "spot_var"

    def title_fragment(self) -> str:
        return (
            f"{self.name}: ξ₀={self.xi0:g}, ω={self.omega:g}, "
            f"κ={self.kappa:g}, ρ={self.rho:g}, X={self.X:g}"
        )

    def as_predict_kwargs(self) -> dict:
        return dict(
            X=self.X,
            r=self.r,
            xi0=self.xi0,
            omega=self.omega,
            kappa=self.kappa,
            rho=self.rho,
            sigma_mode=self.sigma_mode,
            stationary=self.stationary,
        )


DEFAULT_SMILE_SETS: Tuple[SmileParams, ...] = (
    SmileParams("baseline", xi0=0.04, omega=1.0, kappa=2.0, rho=-0.5),
    SmileParams("strong_vovol", xi0=0.04, omega=1.8, kappa=2.0, rho=-0.5),
    SmileParams("strong_neg_skew", xi0=0.04, omega=1.0, kappa=2.0, rho=-0.85),
    SmileParams("mild_fast_mr", xi0=0.04, omega=0.5, kappa=4.0, rho=-0.3),
    SmileParams("slow_mr", xi0=0.04, omega=1.2, kappa=0.6, rho=-0.6),
    SmileParams("near_bs", xi0=0.04, omega=0.05, kappa=2.0, rho=0.0),
)


def error_norms(err: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """L1 / L2 / Linf norms of a grid error (optionally weighted)."""
    e = np.asarray(err, dtype=np.float64).ravel()
    if weights is None:
        w = np.ones_like(e)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        w = w / w.sum()
    abs_e = np.abs(e)
    l1 = float(np.sum(w * abs_e) if weights is not None else abs_e.mean())
    l2 = float(np.sqrt(np.sum(w * abs_e**2) if weights is not None else np.mean(abs_e**2)))
    linf = float(abs_e.max()) if abs_e.size else 0.0
    return {"L1": l1, "L2": l2, "Linf": linf, "MAE": float(abs_e.mean()), "RMSE": l2}


def pinn_price_grid(pinn, S_grid, tau_grid, K: float, smile: SmileParams) -> np.ndarray:
    kwargs = smile.as_predict_kwargs()
    out = np.empty((len(S_grid), len(tau_grid)), dtype=np.float64)
    for i, S in enumerate(S_grid):
        for j, tau in enumerate(tau_grid):
            p = predict_price(pinn, S=float(S), K=K, tau=float(tau), **kwargs)
            out[i, j] = float(p.detach().cpu().reshape(-1)[0])
    return out


def save_heatmap(
    S_grid,
    tau_grid,
    Z,
    title: str,
    path: Path,
    cbar_label: str = "|PINN − QMC+ADD|",
    cmap: str = "magma",
):
    """Heatmap with x=tau, y=S (as requested: (tau, S) plane)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=160)
    # pcolormesh expects Z shaped (len(S), len(tau)) with x=tau, y=S
    Tau, SS = np.meshgrid(tau_grid, S_grid)
    pcm = ax.pcolormesh(Tau, SS, Z, shading="auto", cmap=cmap)
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    ax.set_xlabel("Time to maturity τ")
    ax.set_ylabel("Spot price S")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_fine_grid_price_norms(
    pinn,
    smile: SmileParams,
    out_dir: Path,
    K: float = 100.0,
    s_min: float = 60.0,
    s_max: float = 160.0,
    s_step: float = 2.0,
    tau_min: float = 0.2,
    tau_max: float = 1.6,
    n_tau: int = 29,
    n_paths: int = 131_072,
    n_steps: int = 128,
    seed: int = 42,
) -> dict:
    S_grid = np.arange(s_min, s_max + 0.5 * s_step, s_step, dtype=np.float64)
    tau_grid = np.linspace(tau_min, tau_max, n_tau, dtype=np.float64)

    mc, se = bergomi_qmc_add_grid(
        S_grid=S_grid,
        tau_grid=tau_grid,
        K=K,
        r=smile.r,
        q=smile.q,
        xi0=smile.xi0,
        omega=smile.omega,
        kappa=smile.kappa,
        rho=smile.rho,
        X0=smile.X,
        call_put=getattr(pinn, "call_put", "Call"),
        n_paths=n_paths,
        n_steps=n_steps,
        stationary=smile.stationary,
        seed=seed,
    )
    pinn_p = pinn_price_grid(pinn, S_grid, tau_grid, K, smile)
    abs_err = np.abs(pinn_p - mc)
    rel_err = abs_err / (np.abs(mc) + 1.0)

    abs_norms = error_norms(abs_err)
    rel_norms = error_norms(rel_err)

    stem = f"fine_grid_{smile.name}"
    save_heatmap(
        S_grid,
        tau_grid,
        abs_err,
        title=f"Abs price error |PINN−QMC+ADD|\n{smile.title_fragment()}",
        path=out_dir / f"{stem}_abs_err.png",
        cbar_label="|ΔV|",
    )
    save_heatmap(
        S_grid,
        tau_grid,
        rel_err,
        title=f"Relative price error |PINN−QMC+ADD|/(|MC|+1)\n{smile.title_fragment()}",
        path=out_dir / f"{stem}_rel_err.png",
        cbar_label="rel. error",
        cmap="viridis",
    )

    np.savez_compressed(
        out_dir / f"{stem}.npz",
        S_grid=S_grid,
        tau_grid=tau_grid,
        pinn=pinn_p,
        mc=mc,
        stderr=se,
        abs_err=abs_err,
        rel_err=rel_err,
    )

    return {
        "smile": asdict(smile),
        "grid": {
            "S": [float(S_grid.min()), float(S_grid.max()), float(s_step), int(S_grid.size)],
            "tau": [float(tau_min), float(tau_max), int(n_tau)],
            "n_paths_qmc_add": int(n_paths),
            "n_steps": int(n_steps),
        },
        "abs_norms": abs_norms,
        "rel_norms": rel_norms,
        "mean_mc_stderr": float(np.mean(se)),
        "max_mc_stderr": float(np.max(se)),
        "artifacts": {
            "abs_heatmap": str(out_dir / f"{stem}_abs_err.png"),
            "rel_heatmap": str(out_dir / f"{stem}_rel_err.png"),
            "npz": str(out_dir / f"{stem}.npz"),
        },
    }


def run_smile_heatmaps(
    pinn,
    smiles: Sequence[SmileParams],
    out_dir: Path,
    K: float = 100.0,
    s_min: float = 70.0,
    s_max: float = 140.0,
    s_step: float = 2.5,
    tau_min: float = 0.25,
    tau_max: float = 1.5,
    n_tau: int = 21,
    n_paths: int = 131_072,
    n_steps: int = 128,
    seed: int = 7,
) -> List[dict]:
    results = []
    for k, smile in enumerate(smiles):
        print(f"[smile {k+1}/{len(smiles)}] {smile.title_fragment()}")
        res = run_fine_grid_price_norms(
            pinn,
            smile,
            out_dir=out_dir / "smile_heatmaps",
            K=K,
            s_min=s_min,
            s_max=s_max,
            s_step=s_step,
            tau_min=tau_min,
            tau_max=tau_max,
            n_tau=n_tau,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed + 1000 * k,
        )
        # also emit a dedicated titled copy emphasizing smile params
        data = np.load(res["artifacts"]["npz"])
        save_heatmap(
            data["S_grid"],
            data["tau_grid"],
            data["abs_err"],
            title=(
                f"Price-error norm |PINN − QMC+ADD| on (τ, S)\n"
                f"Smile params — {smile.title_fragment()}"
            ),
            path=out_dir / "smile_heatmaps" / f"heatmap_{smile.name}.png",
            cbar_label="|ΔV| (price norm)",
        )
        res["artifacts"]["smile_heatmap"] = str(
            out_dir / "smile_heatmaps" / f"heatmap_{smile.name}.png"
        )
        results.append(res)
        print(
            f"  abs L1={res['abs_norms']['L1']:.4f}  "
            f"L2={res['abs_norms']['L2']:.4f}  "
            f"Linf={res['abs_norms']['Linf']:.4f}"
        )
    return results


def run_greeks_tests(
    pinn,
    smile: SmileParams,
    out_dir: Path,
    K: float = 100.0,
    spots: Optional[Sequence[float]] = None,
    taus: Optional[Sequence[float]] = None,
    n_paths: int = 262_144,
    n_steps: int = 160,
    seed: int = 123,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if spots is None:
        spots = [70.0, 85.0, 100.0, 115.0, 140.0]
    if taus is None:
        taus = [0.4, 0.8, 1.2, 1.5]

    spots = np.asarray(spots, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    kwargs = smile.as_predict_kwargs()

    delta_pinn = np.zeros((spots.size, taus.size))
    theta_pinn = np.zeros_like(delta_pinn)
    price_pinn = np.zeros_like(delta_pinn)
    delta_mc = np.zeros_like(delta_pinn)
    theta_mc = np.zeros_like(delta_pinn)
    price_mc = np.zeros_like(delta_pinn)

    ss = np.random.SeedSequence(seed)
    child = ss.spawn(spots.size * taus.size)
    idx = 0
    for i, S in enumerate(spots):
        for j, tau in enumerate(taus):
            Vp, Dp, Tp = compute_greeks(
                pinn, S=float(S), K=K, tau=float(tau), **kwargs
            )
            price_pinn[i, j] = float(Vp.detach().cpu().reshape(-1)[0])
            delta_pinn[i, j] = float(Dp.detach().cpu().reshape(-1)[0])
            theta_pinn[i, j] = float(Tp.detach().cpu().reshape(-1)[0])

            pmc, dmc, tmc = Bergomi_QMC_ADD_Greeks(
                S0=float(S),
                K=K,
                T=float(tau),
                r=smile.r,
                q=smile.q,
                xi0=smile.xi0,
                omega=smile.omega,
                kappa=smile.kappa,
                rho=smile.rho,
                X0=smile.X,
                call_put=getattr(pinn, "call_put", "Call"),
                n_paths=n_paths,
                n_steps=n_steps,
                stationary=smile.stationary,
                seed=int(child[idx].generate_state(1)[0]),
            )
            price_mc[i, j] = pmc
            delta_mc[i, j] = dmc
            theta_mc[i, j] = tmc
            idx += 1
            print(
                f"  Greeks S={S:6.1f} τ={tau:4.2f} | "
                f"ΔPINN={delta_pinn[i,j]:+.4f} ΔMC={dmc:+.4f} "
                f"ΘPINN={theta_pinn[i,j]:+.4f} ΘMC={tmc:+.4f}"
            )

    d_err = delta_pinn - delta_mc
    t_err = theta_pinn - theta_mc
    p_err = price_pinn - price_mc

    save_heatmap(
        spots,
        taus,
        np.abs(d_err),
        title=f"|Δ_PINN − Δ_QMC+ADD|\n{smile.title_fragment()}",
        path=out_dir / f"greeks_delta_abs_err_{smile.name}.png",
        cbar_label="|ΔΔ|",
    )
    save_heatmap(
        spots,
        taus,
        np.abs(t_err),
        title=f"|Θ_PINN − Θ_QMC+ADD|\n{smile.title_fragment()}",
        path=out_dir / f"greeks_theta_abs_err_{smile.name}.png",
        cbar_label="|ΔΘ|",
    )

    np.savez_compressed(
        out_dir / f"greeks_{smile.name}.npz",
        spots=spots,
        taus=taus,
        price_pinn=price_pinn,
        price_mc=price_mc,
        delta_pinn=delta_pinn,
        delta_mc=delta_mc,
        theta_pinn=theta_pinn,
        theta_mc=theta_mc,
        delta_err=d_err,
        theta_err=t_err,
        price_err=p_err,
    )

    summary = {
        "smile": asdict(smile),
        "n_paths_qmc_add": int(n_paths),
        "n_steps": int(n_steps),
        "price_norms": error_norms(p_err),
        "delta_norms": error_norms(d_err),
        "theta_norms": error_norms(t_err),
        "artifacts": {
            "spots": spots.tolist(),
            "taus": taus.tolist(),
            "delta_pinn": delta_pinn.tolist(),
            "delta_mc": delta_mc.tolist(),
            "theta_pinn": theta_pinn.tolist(),
            "theta_mc": theta_mc.tolist(),
        },
        "artifacts": {
            "delta_heatmap": str(out_dir / f"greeks_delta_abs_err_{smile.name}.png"),
            "theta_heatmap": str(out_dir / f"greeks_theta_abs_err_{smile.name}.png"),
            "npz": str(out_dir / f"greeks_{smile.name}.npz"),
        },
    }
    return summary


def run_extensive_bergomi_tests(
    checkpoint: str = "trained_models/bergomi.pt",
    out_dir: str | Path = "artifacts/bergomi_extensive_tests",
    device: str = "cpu",
    n_paths_price: int = 131_072,
    n_paths_greeks: int = 262_144,
    n_steps: int = 128,
    smiles: Optional[Sequence[SmileParams]] = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if smiles is None:
        smiles = DEFAULT_SMILE_SETS

    pinn, meta = load_model(checkpoint, device=device)
    pinn.eval()
    print(f"Loaded {checkpoint} | meta sigma_mode={meta.get('sigma_mode')}")

    baseline = smiles[0]
    print("\n=== 1) Fine-grid price norms (baseline smile) ===")
    fine = run_fine_grid_price_norms(
        pinn,
        baseline,
        out_dir=out_dir / "fine_grid",
        n_paths=n_paths_price,
        n_steps=n_steps,
        s_step=2.0,
        n_tau=29,
        seed=42,
    )
    print(
        f"Fine grid abs norms: L1={fine['abs_norms']['L1']:.4f}, "
        f"L2={fine['abs_norms']['L2']:.4f}, Linf={fine['abs_norms']['Linf']:.4f}"
    )

    print("\n=== 2) Smile-structure heatmaps ===")
    smile_results = run_smile_heatmaps(
        pinn,
        smiles,
        out_dir=out_dir,
        n_paths=n_paths_price,
        n_steps=n_steps,
    )

    print("\n=== 3) Greeks PINN vs QMC+ADD ===")
    # Greeks on baseline + strong skew + strong vol-of-vol
    greek_smiles = [smiles[0], smiles[1], smiles[2]]
    greeks_results = []
    for smile in greek_smiles:
        print(f"[greeks] {smile.title_fragment()}")
        greeks_results.append(
            run_greeks_tests(
                pinn,
                smile,
                out_dir=out_dir / "greeks",
                n_paths=n_paths_greeks,
                n_steps=max(n_steps, 160),
            )
        )

    report = {
        "checkpoint": checkpoint,
        "meta": {
            k: meta[k]
            for k in (
                "model_type",
                "sigma_mode",
                "stationary",
                "call_put",
                "hidden",
                "depth",
                "epoch",
                "loss",
            )
            if k in meta
        },
        "method": "scrambled Sobol QMC + antithetic differencing (ADD)",
        "fine_grid": fine,
        "smile_heatmaps": smile_results,
        "greeks": greeks_results,
    }

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")
    return report
