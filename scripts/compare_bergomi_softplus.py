#!/usr/bin/env python3
"""A/B: bergomi.pt vs softplus-finetuned (+ inference-only softplus wrap)."""

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
    out = Path("artifacts/bergomi_softplus_ab")
    out.mkdir(parents=True, exist_ok=True)
    opt = Path("/opt/cursor/artifacts/bergomi_softplus_ab")
    opt.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    smile = SmileParams("baseline", xi0=0.04, omega=1.0, kappa=2.0, rho=-0.5)
    K = 100.0
    kwargs = smile.as_predict_kwargs()

    pinn_old, meta_old = load_model("trained_models/bergomi.pt", device=device)
    pinn_new, meta_new = load_model(
        "trained_models/bergomi_softplus.pt", device=device
    )
    # Inference-only wrap: same weights as OLD, softplus enabled at eval.
    pinn_wrap, _ = load_model("trained_models/bergomi.pt", device=device)
    pinn_wrap.price_softplus_beta = float(
        meta_new.get("price_softplus_beta", 40.0)
    )

    for m in (pinn_old, pinn_new, pinn_wrap):
        m.eval()

    print(
        "OLD β",
        getattr(pinn_old, "price_softplus_beta", None),
        "| NEW β",
        getattr(pinn_new, "price_softplus_beta", None),
        "loss",
        meta_new.get("loss"),
        "epoch",
        meta_new.get("epoch"),
        "| WRAP β",
        pinn_wrap.price_softplus_beta,
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

    grids = {
        "old": price_grid(pinn_old, S_grid, tau_grid, K, kwargs),
        "wrap": price_grid(pinn_wrap, S_grid, tau_grid, K, kwargs),
        "new": price_grid(pinn_new, S_grid, tau_grid, K, kwargs),
    }
    norms = {k: error_norms(np.abs(v - mc)) for k, v in grids.items()}
    for k, n in norms.items():
        print(f"{k.upper()} abs norms", n)
        print(
            f"  neg={int((grids[k] < 0).sum())} min={float(grids[k].min()):.6f}"
        )
    print(
        "L1 OLD→NEW",
        norms["old"]["L1"] - norms["new"]["L1"],
        f"({100 * (norms['old']['L1'] - norms['new']['L1']) / norms['old']['L1']:.1f}%)",
    )
    print(
        "L1 OLD→WRAP",
        norms["old"]["L1"] - norms["wrap"]["L1"],
        f"({100 * (norms['old']['L1'] - norms['wrap']['L1']) / norms['old']['L1']:.1f}%)",
    )

    save_heatmap(
        S_grid,
        tau_grid,
        np.abs(grids["old"] - mc),
        title="OLD |PINN−QMC+ADD|\nno softplus",
        path=out / "err_old.png",
        cbar_label="|ΔV|",
    )
    save_heatmap(
        S_grid,
        tau_grid,
        np.abs(grids["new"] - mc),
        title="SOFTPLUS-FT |PINN−QMC+ADD|\nu=softplus_β(u_BS+U)",
        path=out / "err_new.png",
        cbar_label="|ΔV|",
    )
    save_heatmap(
        S_grid,
        tau_grid,
        np.abs(grids["old"] - mc) - np.abs(grids["new"] - mc),
        title="OLD−SOFTPLUS-FT abs error (positive = improved)",
        path=out / "err_old_minus_new.png",
        cbar_label="Δ|err|",
        cmap="coolwarm",
    )
    save_heatmap(
        S_grid,
        tau_grid,
        np.abs(grids["wrap"] - mc),
        title="SOFTPLUS-WRAP |PINN−QMC+ADD|\nOLD weights + softplus@eval",
        path=out / "err_wrap.png",
        cbar_label="|ΔV|",
    )

    print("\nGreeks A/B vs QMC+ADD", flush=True)
    grows = []
    for S in [70.0, 85.0, 100.0, 115.0, 140.0]:
        for tau in [0.4, 1.0, 1.4]:
            Vo, Do, _ = [
                float(x)
                for x in compute_greeks(pinn_old, S=S, K=K, tau=tau, **kwargs)
            ]
            Vn, Dn, _ = [
                float(x)
                for x in compute_greeks(pinn_new, S=S, K=K, tau=tau, **kwargs)
            ]
            pmc, dmc, _ = Bergomi_QMC_ADD_Greeks(
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
                f"|Δ| old={abs(Do-dmc):.4f} new={abs(Dn-dmc):.4f} Vnew={Vn:.4f}"
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
        ew = error_norms(np.abs(price_grid(pinn_wrap, S2, tau2, K, kw) - mc2))
        extra.append(dict(name=sm.name, old=eo, new=en, wrap=ew))
        print(
            sm.name,
            "L1 old",
            eo["L1"],
            "new",
            en["L1"],
            "wrap",
            ew["L1"],
            "dL1",
            eo["L1"] - en["L1"],
        )

    report = dict(
        note=(
            "Positivity uses softplus_β(u)=log(1+e^{βu})/β (scalar). "
            "Softmax is not applicable to a single price."
        ),
        train=dict(
            best_loss=meta_new.get("loss"),
            best_epoch=meta_new.get("epoch"),
            price_softplus_beta=meta_new.get("price_softplus_beta"),
            positivity=meta_new.get("positivity"),
        ),
        baseline_smile=dict(
            old=norms["old"],
            wrap=norms["wrap"],
            new=norms["new"],
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
            old=int((grids["old"] < 0).sum()),
            wrap=int((grids["wrap"] < 0).sum()),
            new=int((grids["new"] < 0).sum()),
            old_min=float(grids["old"].min()),
            wrap_min=float(grids["wrap"].min()),
            new_min=float(grids["new"].min()),
        ),
    )
    (out / "comparison_report.json").write_text(json.dumps(report, indent=2))
    (opt / "comparison_report.json").write_text(json.dumps(report, indent=2))
    for png in out.glob("*.png"):
        shutil.copy(png, opt / png.name)
    print("\nDONE", out)


if __name__ == "__main__":
    main()
