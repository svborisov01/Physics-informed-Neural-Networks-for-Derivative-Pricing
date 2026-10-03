#!/usr/bin/env python3
"""
Fine-tune Bergomi PINN with hybrid PDE + sparse QMC+ADD supervision.

Starts from trained_models/bergomi.pt (tight correction head) and adds a small
set of QMC+ADD price labels across a few smile regimes.
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

from pricing.bergomi_option_pricing import train_network
from support_tools.bergomi_test_suite import DEFAULT_SMILE_SETS
from support_tools.model_wrapper import load_model
from support_tools.monte_carlo_pricing_tools import bergomi_qmc_add_grid


def build_qmc_supervision_dataset(
    smiles,
    K: float = 100.0,
    S_grid=None,
    tau_grid=None,
    n_paths: int = 65_536,
    n_steps: int = 96,
    seed: int = 123,
    atm_weight_scale: float = 0.20,
):
    """
    Build a compact labeled set: a few smiles × ATM-biased (S, τ) grid.

    Returns a dict of float32 tensors ready for ``train_network``.
    """
    if S_grid is None:
        S_grid = np.array(
            [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 140.0],
            dtype=np.float64,
        )
    if tau_grid is None:
        tau_grid = np.array(
            [0.25, 0.5, 0.75, 1.0, 1.25, 1.5], dtype=np.float64
        )

    xs, Xs, taus, rs, xi0s, omegas, kappas, rhos = [], [], [], [], [], [], [], []
    u_tgts, weights, names = [], [], []

    for i, sm in enumerate(smiles):
        print(
            f"  QMC labels [{i+1}/{len(smiles)}] {sm.name} "
            f"(n_paths={n_paths})…",
            flush=True,
        )
        prices, stderrs = bergomi_qmc_add_grid(
            S_grid,
            tau_grid,
            K=K,
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
            seed=seed + 17 * i,
        )
        for j, S in enumerate(S_grid):
            x = float(np.log(S / K))
            # Emphasize ATM / near-ATM; still keep wings with lower weight.
            atm_w = float(np.exp(-0.5 * (x / atm_weight_scale) ** 2))
            for k, tau in enumerate(tau_grid):
                se = float(stderrs[j, k])
                # Down-weight noisy labels.
                se_w = 1.0 / (1.0 + (se / max(K * 1e-3, 1e-6)) ** 2)
                w = (0.35 + 0.65 * atm_w) * se_w
                xs.append(x)
                Xs.append(float(sm.X))
                taus.append(float(tau))
                rs.append(float(sm.r))
                xi0s.append(float(sm.xi0))
                omegas.append(float(sm.omega))
                kappas.append(float(sm.kappa))
                rhos.append(float(sm.rho))
                u_tgts.append(float(prices[j, k] / K))
                weights.append(w)
                names.append(sm.name)

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
        u_target=_t(u_tgts),
        weight=_t(weights),
    )
    meta = dict(
        n_points=int(data["u_target"].shape[0]),
        K=K,
        S_grid=S_grid.tolist(),
        tau_grid=tau_grid.tolist(),
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        smiles=[s.name for s in smiles],
        u_target_mean=float(np.mean(u_tgts)),
        weight_mean=float(np.mean(weights)),
    )
    return data, meta, names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--init-checkpoint",
        default="trained_models/bergomi.pt",
        help="PDE-only checkpoint to fine-tune",
    )
    p.add_argument(
        "--out-checkpoint",
        default="trained_models/bergomi_hybrid.pt",
    )
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-data", type=float, default=5.0)
    p.add_argument("--N-pde", type=int, default=4000)
    p.add_argument("--N-boundary", type=int, default=1500)
    p.add_argument("--N-data", type=int, default=256)
    p.add_argument("--n-paths", type=int, default=65_536)
    p.add_argument("--n-steps", type=int, default=96)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    out_ckpt = Path(args.out_checkpoint)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    art = Path("artifacts/bergomi_hybrid")
    art.mkdir(parents=True, exist_ok=True)
    opt = Path("/opt/cursor/artifacts/bergomi_hybrid")
    opt.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading {args.init_checkpoint} on {device}…", flush=True)
    pinn, meta = load_model(args.init_checkpoint, device=device)
    pinn.train()

    smiles = list(DEFAULT_SMILE_SETS)
    # Keep near_bs so the hybrid term does not drift the ω→0 limit.
    print(f"Building supervised set ({len(smiles)} smiles)…", flush=True)
    data, data_meta, _ = build_qmc_supervision_dataset(
        smiles,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        seed=123,
    )
    (art / "supervised_meta.json").write_text(json.dumps(data_meta, indent=2))
    (opt / "supervised_meta.json").write_text(json.dumps(data_meta, indent=2))
    print(f"Supervised points: {data_meta['n_points']}", flush=True)

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
        supervised_data=data,
        N_data=args.N_data,
        detach_source=True,
        grad_clip=1.0,
        print_every=args.print_every,
        best_model_path=str(out_ckpt),
        save_model=True,
        weight_decay=args.weight_decay,
        cosine_eta_min=1e-6,
        cosine_T0=400,
        extra_checkpoint_meta=dict(
            hybrid_init=str(args.init_checkpoint),
            supervised_meta=data_meta,
            train_lr=args.lr,
            train_weight_decay=args.weight_decay,
            train_epochs=args.epochs,
        ),
    )

    hist_path = art / "train_history.json"
    # Keep history JSON light (drop huge per-epoch lists if needed — keep all for now).
    serializable = {
        k: (v if not isinstance(v, list) else v)
        for k, v in history.items()
    }
    hist_path.write_text(json.dumps(serializable, indent=2))
    (opt / "train_history.json").write_text(hist_path.read_text())
    print(
        f"DONE best_loss={history['best_loss']:.6e} @ {history['best_epoch']} "
        f"→ {out_ckpt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
