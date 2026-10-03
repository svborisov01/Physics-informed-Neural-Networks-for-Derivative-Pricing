#!/usr/bin/env python3
"""A/B compare bergomi.pt vs bergomi_hybrid_v3.pt (and optional v2)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from support_tools.bergomi_test_suite import SmileParams, error_norms, save_heatmap
from support_tools.model_wrapper import compute_greeks, load_model, predict_price
from support_tools.monte_carlo_pricing_tools import (
    Bergomi_QMC_ADD_Greeks,
    bergomi_qmc_add_grid,
)


def price_grid(pinn, S_grid, tau_grid, K, kwargs):
    Z = np.zeros((len(S_grid), len(tau_grid)))
    for i, S in enumerate(S_grid):
        for j, tau in enumerate(tau_grid):
            Z[i, j] = float(
                predict_price(
                    pinn, S=float(S), K=K, tau=float(tau), **kwargs
                )
                .detach()
                .cpu()
            )
    return Z


def main():
    out = Path("artifacts/bergomi_hybrid_v3_ab")
    out.mkdir(parents=True, exist_ok=True)
    opt = Path("/opt/cursor/artifacts/bergomi_hybrid_v3_ab")
    opt.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    smile = SmileParams("baseline", xi0=0.04, omega=1.0, kappa=2.0, rho=-0.5)
    K = 100.0
    kwargs = smile.as_predict_kwargs()

    pinn_old, meta_old = load_model("trained_models/bergomi.pt", device=device)
    pinn_new, meta_new = load_model(
        "trained_models/bergomi_hybrid_v3.pt", device=device
    )
    pinn_old.eval()
    pinn_new.eval()

    has_v2 = Path("trained_models/bergomi_hybrid.pt").is_file()
    pinn_v2 = meta_v2 = None
    if has_v2:
        pinn_v2, meta_v2 = load_model(
            "trained_models/bergomi_hybrid.pt", device=device
        )
        pinn_v2.eval()

    print(
        "Loaded | old",
        meta_old.get("loss"),
        "| v3",
        meta_new.get("loss"),
        "val_l1",
        meta_new.get("val_l1"),
        "epoch",
        meta_new.get("epoch"),
        "λd/Δ/otm",
        meta_new.get("lambda_data"),
        meta_new.get("lambda_delta"),
        meta_new.get("lambda_otm"),
    )

    S_grid = np.arange(60.0, 160.0 + 1e-9, 2.0)
    tau_grid = np.linspace(0.2, 1.6, 29)
    print("QMC+ADD reference…", flush=True)
    mc, se = bergomi_qmc_add_grid(
        S_grid,
        tau_grid,
        K=K,
        r=smile.r,
        q=0.0,
        xi0=smile.xi0,
        omega=smile.omega,
        kappa=smile.kappa,
        rho=smile.rho,
        X0=0.0,
        n_paths=131_072,
        n_steps=128,
        stationary=True,
        seed=42,
    )
    old_p = price_grid(pinn_old, S_grid, tau_grid, K, kwargs)
    new_p = price_grid(pinn_new, S_grid, tau_grid, K, kwargs)
    old_n = error_norms(np.abs(old_p - mc))
    new_n = error_norms(np.abs(new_p - mc))
    print("OLD abs norms", old_n)
    print("V3 abs norms", new_n)
    print(
        "L1 improvement vs OLD",
        old_n["L1"] - new_n["L1"],
        f"({100 * (old_n['L1'] - new_n['L1']) / old_n['L1']:.1f}%)",
    )
    print(
        "neg prices OLD",
        int((old_p < 0).sum()),
        "V3",
        int((new_p < 0).sum()),
        "min V3",
        float(new_p.min()),
    )

    v2_n = None
    if pinn_v2 is not None:
        v2_p = price_grid(pinn_v2, S_grid, tau_grid, K, kwargs)
        v2_n = error_norms(np.abs(v2_p - mc))
        print("V2 abs norms", v2_n)
        print(
            "L1 improvement v3 vs v2",
            v2_n["L1"] - new_n["L1"],
            f"({100 * (v2_n['L1'] - new_n['L1']) / v2_n['L1']:.1f}%)",
        )

    save_heatmap(
        S_grid,
        tau_grid,
        np.abs(old_p - mc),
        title="OLD |PINN−QMC+ADD|\nbaseline smile",
        path=out / "err_old.png",
        cbar_label="|ΔV|",
    )
    save_heatmap(
        S_grid,
        tau_grid,
        np.abs(new_p - mc),
        title="HYBRID v3 |PINN−QMC+ADD|\nOTM hinge + holdout + Δ labels",
        path=out / "err_new.png",
        cbar_label="|ΔV|",
    )
    save_heatmap(
        S_grid,
        tau_grid,
        np.abs(old_p - mc) - np.abs(new_p - mc),
        title="OLD−v3 abs error (positive = improved)",
        path=out / "err_old_minus_new.png",
        cbar_label="Δ|err|",
        cmap="coolwarm",
    )

    print("\nGreeks A/B vs QMC+ADD", flush=True)
    grows = []
    for S in [70.0, 85.0, 100.0, 115.0, 140.0]:
        for tau in [0.4, 1.0, 1.4]:
            Vo, Do, To = [
                float(x)
                for x in compute_greeks(pinn_old, S=S, K=K, tau=tau, **kwargs)
            ]
            Vn, Dn, Tn = [
                float(x)
                for x in compute_greeks(pinn_new, S=S, K=K, tau=tau, **kwargs)
            ]
            pmc, dmc, tmc = Bergomi_QMC_ADD_Greeks(
                S0=S,
                K=K,
                T=tau,
                r=0.05,
                q=0.0,
                xi0=0.04,
                omega=1.0,
                kappa=2.0,
                rho=-0.5,
                n_paths=131_072,
                n_steps=160,
                seed=abs(hash((S, tau))) % (2**31),
            )
            grows.append(
                dict(
                    S=S,
                    tau=tau,
                    V_old=Vo,
                    V_new=Vn,
                    V_mc=pmc,
                    d_old=Do,
                    d_new=Dn,
                    d_mc=dmc,
                    d_err_old=abs(Do - dmc),
                    d_err_new=abs(Dn - dmc),
                    v_err_old=abs(Vo - pmc),
                    v_err_new=abs(Vn - pmc),
                )
            )
            print(
                f"S={S:5.1f} τ={tau:.1f} |V| old={abs(Vo-pmc):.3f} new={abs(Vn-pmc):.3f} "
                f"|Δ| old={abs(Do-dmc):.4f} new={abs(Dn-dmc):.4f}"
            )

    d_old = float(np.mean([g["d_err_old"] for g in grows]))
    d_new = float(np.mean([g["d_err_new"] for g in grows]))
    v_old = float(np.mean([g["v_err_old"] for g in grows]))
    v_new = float(np.mean([g["v_err_new"] for g in grows]))
    print(f"\nMean |V| err old={v_old:.4f} new={v_new:.4f}")
    print(f"Mean |Δ| err old={d_old:.4f} new={d_new:.4f}")

    print("\nExtra smile L1", flush=True)
    extra = []
    S2 = np.arange(70, 141, 5.0)
    tau2 = np.linspace(0.25, 1.5, 13)
    for sm in [
        SmileParams("strong_vovol", 0.04, 1.8, 2.0, -0.5),
        SmileParams("strong_neg_skew", 0.04, 1.0, 2.0, -0.85),
        SmileParams("slow_mr", 0.04, 1.2, 0.6, -0.6),
        SmileParams("near_bs", 0.04, 0.05, 2.0, 0.0),
    ]:
        kw = sm.as_predict_kwargs()
        mc2, _ = bergomi_qmc_add_grid(
            S2,
            tau2,
            K=K,
            r=sm.r,
            q=0.0,
            xi0=sm.xi0,
            omega=sm.omega,
            kappa=sm.kappa,
            rho=sm.rho,
            n_paths=65_536,
            n_steps=96,
            seed=7,
        )
        eo = error_norms(np.abs(price_grid(pinn_old, S2, tau2, K, kw) - mc2))
        en = error_norms(np.abs(price_grid(pinn_new, S2, tau2, K, kw) - mc2))
        row = dict(name=sm.name, old=eo, new=en)
        if pinn_v2 is not None:
            ev2 = error_norms(np.abs(price_grid(pinn_v2, S2, tau2, K, kw) - mc2))
            row["v2"] = ev2
        extra.append(row)
        print(sm.name, "L1 old", eo["L1"], "v3", en["L1"], "dL1", eo["L1"] - en["L1"])

    report = dict(
        train=dict(
            best_loss=meta_new.get("loss"),
            best_epoch=meta_new.get("epoch"),
            val_l1=meta_new.get("val_l1"),
            lambda_data=meta_new.get("lambda_data"),
            lambda_delta=meta_new.get("lambda_delta"),
            lambda_otm=meta_new.get("lambda_otm"),
            hybrid_version=meta_new.get("hybrid_version"),
            supervised_meta=meta_new.get("supervised_meta"),
            v2_loss=None if meta_v2 is None else meta_v2.get("loss"),
        ),
        baseline_smile=dict(
            old=old_n,
            new=new_n,
            v2=v2_n,
            mean_mc_stderr=float(np.mean(se)),
        ),
        greek_summary=dict(
            mean_abs_V_old=v_old,
            mean_abs_V_new=v_new,
            mean_abs_delta_old=d_old,
            mean_abs_delta_new=d_new,
        ),
        greeks_points=grows,
        extra_smiles=extra,
        neg_prices=dict(
            old=int((old_p < 0).sum()),
            new=int((new_p < 0).sum()),
            old_min=float(old_p.min()),
            new_min=float(new_p.min()),
        ),
    )
    (out / "comparison_report.json").write_text(json.dumps(report, indent=2))
    (opt / "comparison_report.json").write_text(json.dumps(report, indent=2))
    for png in out.glob("*.png"):
        shutil.copy(png, opt / png.name)
    print("\nDONE", out)


if __name__ == "__main__":
    main()
