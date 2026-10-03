#!/usr/bin/env python3
"""
Hybrid Bergomi v3 fine-tune:
  1. OTM negativity hinge (+ light low-strike Delta BC)
  2. Hold out baseline smile for val-L1 checkpointing / early stop
  3. Sparse ATM pathwise Delta QMC+ADD labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

from pricing.bergomi_option_pricing import train_network
from support_tools.bergomi_test_suite import DEFAULT_SMILE_SETS
from support_tools.model_wrapper import load_model
from support_tools.monte_carlo_pricing_tools import Bergomi_QMC_ADD_Greeks

_spec = importlib.util.spec_from_file_location(
    "train_bergomi_hybrid",
    _ROOT / "scripts" / "train_bergomi_hybrid.py",
)
_hybrid_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_hybrid_mod)
build_qmc_supervision_dataset = _hybrid_mod.build_qmc_supervision_dataset


def build_atm_delta_dataset(
    smiles,
    K: float = 100.0,
    S_list=None,
    tau_list=None,
    n_paths: int = 65_536,
    n_steps: int = 128,
    seed: int = 321,
):
    """Small ATM-focused pathwise Delta label set."""
    if S_list is None:
        S_list = [90.0, 95.0, 100.0, 105.0, 110.0]
    if tau_list is None:
        tau_list = [0.4, 0.75, 1.0, 1.4]

    xs, Xs, taus, rs, xi0s, omegas, kappas, rhos = [], [], [], [], [], [], [], []
    deltas, weights = [], []

    n_jobs = len(smiles) * len(S_list) * len(tau_list)
    done = 0
    for i, sm in enumerate(smiles):
        for S in S_list:
            for tau in tau_list:
                done += 1
                if done % 10 == 1 or done == n_jobs:
                    print(
                        f"  Delta labels {done}/{n_jobs} "
                        f"({sm.name}, S={S}, τ={tau})…",
                        flush=True,
                    )
                _price, delta, _theta = Bergomi_QMC_ADD_Greeks(
                    S0=float(S),
                    K=K,
                    T=float(tau),
                    r=sm.r,
                    q=sm.q,
                    xi0=sm.xi0,
                    omega=sm.omega,
                    kappa=sm.kappa,
                    rho=sm.rho,
                    X0=sm.X,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    stationary=sm.stationary,
                    seed=seed + 31 * i + int(10 * S) + int(100 * tau),
                )
                x = float(np.log(S / K))
                # Slightly higher weight exactly ATM.
                w = float(np.exp(-0.5 * (x / 0.12) ** 2))
                xs.append(x)
                Xs.append(float(sm.X))
                taus.append(float(tau))
                rs.append(float(sm.r))
                xi0s.append(float(sm.xi0))
                omegas.append(float(sm.omega))
                kappas.append(float(sm.kappa))
                rhos.append(float(sm.rho))
                deltas.append(float(delta))
                weights.append(0.5 + 0.5 * w)

    def _t(a):
        return torch.tensor(np.asarray(a, dtype=np.float32)).view(-1, 1)

    data = dict(
        x=_t(xs),
        X=_t(Xs),
        tau=_t(taus),
        r=_t(rs),
        xi0=_t(xi0s),
        omega=_t(omegas),
        kappa=_t(kappas),
        rho=_t(rhos),
        delta_target=_t(deltas),
        weight=_t(weights),
    )
    meta = dict(
        n_points=int(data["delta_target"].shape[0]),
        K=K,
        S_list=list(S_list),
        tau_list=list(tau_list),
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        smiles=[s.name for s in smiles],
        delta_mean=float(np.mean(deltas)),
    )
    return data, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--init-checkpoint",
        default="trained_models/bergomi_hybrid.pt",
        help="Start from hybrid v2 if present, else bergomi.pt",
    )
    p.add_argument(
        "--out-checkpoint",
        default="trained_models/bergomi_hybrid_v3.pt",
    )
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-data", type=float, default=25.0)
    p.add_argument("--lambda-delta", type=float, default=5.0)
    p.add_argument("--lambda-otm", type=float, default=2.0)
    p.add_argument("--N-pde", type=int, default=3500)
    p.add_argument("--N-boundary", type=int, default=1200)
    p.add_argument("--N-data", type=int, default=256)
    p.add_argument("--N-delta", type=int, default=48)
    p.add_argument("--N-otm", type=int, default=1500)
    p.add_argument("--n-paths", type=int, default=65_536)
    p.add_argument("--n-steps", type=int, default=96)
    p.add_argument("--delta-paths", type=int, default=65_536)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument(
        "--holdout-smile",
        default="baseline",
        help="Smile name held out for val-L1 selection",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    init_ckpt = Path(args.init_checkpoint)
    if not init_ckpt.is_file():
        init_ckpt = Path("trained_models/bergomi.pt")
        print(f"Init missing; falling back to {init_ckpt}", flush=True)

    out_ckpt = Path(args.out_checkpoint)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    art = Path("artifacts/bergomi_hybrid_v3")
    art.mkdir(parents=True, exist_ok=True)
    opt = Path("/opt/cursor/artifacts/bergomi_hybrid_v3")
    opt.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading {init_ckpt} on {device}…", flush=True)
    pinn, meta = load_model(str(init_ckpt), device=device)
    pinn.train()

    all_smiles = list(DEFAULT_SMILE_SETS)
    holdout = args.holdout_smile
    train_smiles = [s for s in all_smiles if s.name != holdout]
    val_smiles = [s for s in all_smiles if s.name == holdout]
    if not train_smiles or not val_smiles:
        raise SystemExit(f"holdout smile {holdout!r} not found in DEFAULT_SMILE_SETS")

    print(
        f"Train smiles: {[s.name for s in train_smiles]} | "
        f"val holdout: {[s.name for s in val_smiles]}",
        flush=True,
    )
    print("Building train price labels…", flush=True)
    train_data, train_meta, _ = build_qmc_supervision_dataset(
        train_smiles,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        seed=123,
    )
    print("Building val (holdout) price labels…", flush=True)
    val_data, val_meta, _ = build_qmc_supervision_dataset(
        val_smiles,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        seed=999,
    )
    print("Building ATM Delta labels (train smiles only)…", flush=True)
    delta_data, delta_meta = build_atm_delta_dataset(
        train_smiles,
        n_paths=args.delta_paths,
        n_steps=max(args.n_steps, 128),
        seed=321,
    )

    meta_all = dict(
        train=train_meta,
        val=val_meta,
        delta=delta_meta,
        holdout_smile=holdout,
        init_checkpoint=str(init_ckpt),
        lambda_data=args.lambda_data,
        lambda_delta=args.lambda_delta,
        lambda_otm=args.lambda_otm,
    )
    (art / "supervised_meta.json").write_text(json.dumps(meta_all, indent=2))
    (opt / "supervised_meta.json").write_text(json.dumps(meta_all, indent=2))
    print(
        f"Sizes: train={train_meta['n_points']} val={val_meta['n_points']} "
        f"delta={delta_meta['n_points']}",
        flush=True,
    )

    history = train_network(
        pinn,
        N_pde=args.N_pde,
        N_boundary=args.N_boundary,
        epochs=args.epochs,
        lr=args.lr,
        sigma_mode=meta.get("sigma_mode", "spot_var"),
        stationary=bool(meta.get("stationary", True)),
        lambda_boundary=1.0,
        lambda_terminal=1.0,
        lambda_data=args.lambda_data,
        lambda_delta=args.lambda_delta,
        lambda_otm=args.lambda_otm,
        supervised_data=train_data,
        supervised_delta_data=delta_data,
        val_data=val_data,
        N_data=args.N_data,
        N_delta=args.N_delta,
        N_otm=args.N_otm,
        detach_source=True,
        grad_clip=1.0,
        print_every=args.print_every,
        best_model_path=str(out_ckpt),
        save_model=True,
        weight_decay=args.weight_decay,
        cosine_eta_min=1e-6,
        cosine_T0=400,
        selection_metric="val_l1",
        early_stop_patience=args.patience,
        extra_checkpoint_meta=dict(
            hybrid_version="v3",
            hybrid_init=str(init_ckpt),
            supervised_meta=meta_all,
            train_lr=args.lr,
            train_weight_decay=args.weight_decay,
            train_epochs=args.epochs,
        ),
    )

    hist_path = art / "train_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    (opt / "train_history.json").write_text(hist_path.read_text())
    print(
        f"DONE best_val_l1={history.get('best_val_l1')} @ {history['best_epoch']} "
        f"→ {out_ckpt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
