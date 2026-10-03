#!/usr/bin/env python3
"""A/B compare baseline bergomi.pt vs bergomi_improved.pt on QMC+ADD."""

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
    out = Path("artifacts/bergomi_improvement_ab")
    out.mkdir(parents=True, exist_ok=True)
    opt = Path("/opt/cursor/artifacts/bergomi_improvement_ab")
    opt.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    smile = SmileParams("baseline", xi0=0.04, omega=1.0, kappa=2.0, rho=-0.5)
    K = 100.0
    kwargs = smile.as_predict_kwargs()

    pinn_old, meta_old = load_model("trained_models/bergomi.pt", device=device)
    pinn_new, meta_new = load_model(
        "trained_models/bergomi_improved.pt", device=device
    )
    pinn_old.eval()
    pinn_new.eval()
    meta_keys = (
        "loss",
        "epoch",
        "corr_scale0",
        "omega_gate_power",
        "price_softplus_beta",
        "corr_activation",
    )
    print(
        "OLD head",
        {
            "corr_scale0": getattr(pinn_old, "corr_scale0", None),
            "omega_gate_power": getattr(pinn_old, "omega_gate_power", None),
            "price_softplus_beta": getattr(pinn_old, "price_softplus_beta", None),
            "corr_activation": getattr(pinn_old, "corr_activation", None),
            "ckpt_loss": meta_old.get("loss"),
        },
    )
    print(
        "NEW head",
        {k: meta_new.get(k) for k in meta_keys}
        | {
            "corr_activation": getattr(pinn_new, "corr_activation", None),
        },
    )

    S_grid = np.arange(60.0, 160.0 + 1e-9, 2.0)
    tau_grid = np.linspace(0.2, 1.6, 29)
    print("QMC+ADD reference…")
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
    old_err = np.abs(old_p - mc)
    new_err = np.abs(new_p - mc)
    old_n = error_norms(old_err)
    new_n = error_norms(new_err)
    print("OLD abs norms", old_n)
    print("NEW abs norms", new_n)
    print(
        "L1 improvement",
        old_n["L1"] - new_n["L1"],
        f"({100 * (old_n['L1'] - new_n['L1']) / old_n['L1']:.1f}%)",
    )
    print("neg prices OLD", int((old_p < 0).sum()), "NEW", int((new_p < 0).sum()), "min NEW", float(new_p.min()))

    save_heatmap(
        S_grid,
        tau_grid,
        old_err,
        title="OLD |PINN−QMC+ADD|\nbaseline smile",
        path=out / "err_old.png",
        cbar_label="|ΔV|",
    )
    save_heatmap(
        S_grid,
        tau_grid,
        new_err,
        title="NEW |PINN−QMC+ADD|\nfreer head + adaptive + softplus",
        path=out / "err_new.png",
        cbar_label="|ΔV|",
    )
    save_heatmap(
        S_grid,
        tau_grid,
        old_err - new_err,
        title="OLD−NEW abs error (positive = improved)",
        path=out / "err_old_minus_new.png",
        cbar_label="Δ|err|",
        cmap="coolwarm",
    )

    print("\nGreeks A/B vs QMC+ADD")
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
                    t_old=To,
                    t_new=Tn,
                    t_mc=tmc,
                    d_err_old=abs(Do - dmc),
                    d_err_new=abs(Dn - dmc),
                    t_err_old=abs(To - tmc),
                    t_err_new=abs(Tn - tmc),
                    v_err_old=abs(Vo - pmc),
                    v_err_new=abs(Vn - pmc),
                )
            )
            print(
                f"S={S:5.1f} τ={tau:.1f} |V| old={abs(Vo-pmc):.3f} new={abs(Vn-pmc):.3f} "
                f"|Δ| old={abs(Do-dmc):.4f} new={abs(Dn-dmc):.4f} Vnew={Vn:.3f}"
            )

    d_old = float(np.mean([g["d_err_old"] for g in grows]))
    d_new = float(np.mean([g["d_err_new"] for g in grows]))
    v_old = float(np.mean([g["v_err_old"] for g in grows]))
    v_new = float(np.mean([g["v_err_new"] for g in grows]))
    print(f"\nMean |V| err old={v_old:.4f} new={v_new:.4f}")
    print(f"Mean |Δ| err old={d_old:.4f} new={d_new:.4f}")

    print("\nExtra smile L1")
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
        extra.append(dict(name=sm.name, old=eo, new=en))
        print(sm.name, "L1 old", eo["L1"], "new", en["L1"], "dL1", eo["L1"] - en["L1"])

    report = dict(
        train=dict(
            best_loss=meta_new.get("loss"),
            best_epoch=meta_new.get("epoch"),
            epochs=3000,
            corr_scale0=meta_new.get("corr_scale0"),
            omega_gate_power=meta_new.get("omega_gate_power"),
            price_softplus_beta=meta_new.get("price_softplus_beta"),
        ),
        baseline_smile=dict(old=old_n, new=new_n, mean_mc_stderr=float(np.mean(se))),
        greeks_points=grows,
        greek_summary=dict(
            mean_abs_V_old=v_old,
            mean_abs_V_new=v_new,
            mean_abs_delta_old=d_old,
            mean_abs_delta_new=d_new,
        ),
        extra_smiles=extra,
        neg_prices=dict(
            old=int((old_p < 0).sum()),
            new=int((new_p < 0).sum()),
            new_min=float(new_p.min()),
            old_min=float(old_p.min()),
        ),
    )
    (out / "comparison_report.json").write_text(json.dumps(report, indent=2))
    (opt / "comparison_report.json").write_text(json.dumps(report, indent=2))
    for p in out.glob("*.png"):
        shutil.copy(p, opt / p.name)
    print("\nDONE", out)


if __name__ == "__main__":
    main()
