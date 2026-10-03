#!/usr/bin/env python3
"""
Fine-tune Bergomi PINN with softplus positivity on u = softplus(u_BS + U).

Starts from trained_models/bergomi.pt (tight correction head). Softplus — not
softmax — is the scalar positivity map that eliminates negative prices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pricing.bergomi_option_pricing import train_network
from support_tools.model_wrapper import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-checkpoint", default="trained_models/bergomi.pt")
    p.add_argument(
        "--out-checkpoint", default="trained_models/bergomi_softplus.pt"
    )
    p.add_argument("--softplus-beta", type=float, default=400.0)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--N-pde", type=int, default=4000)
    p.add_argument("--N-boundary", type=int, default=1500)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    out = Path(args.out_checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    art = Path("artifacts/bergomi_softplus")
    art.mkdir(parents=True, exist_ok=True)
    opt = Path("/opt/cursor/artifacts/bergomi_softplus")
    opt.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading {args.init_checkpoint} on {device}…", flush=True)
    pinn, meta = load_model(args.init_checkpoint, device=device)
    pinn.price_softplus_beta = float(args.softplus_beta)
    pinn.train()
    print(f"Enabled softplus β={pinn.price_softplus_beta}", flush=True)

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
        lambda_data=0.0,
        detach_source=True,
        grad_clip=1.0,
        print_every=args.print_every,
        best_model_path=str(out),
        save_model=True,
        weight_decay=args.weight_decay,
        cosine_eta_min=1e-6,
        cosine_T0=400,
        selection_metric="total",
        extra_checkpoint_meta=dict(
            softplus_init=str(args.init_checkpoint),
            price_softplus_beta=float(args.softplus_beta),
            train_lr=args.lr,
            train_weight_decay=args.weight_decay,
            train_epochs=args.epochs,
            positivity="softplus",
        ),
    )

    (art / "train_history.json").write_text(json.dumps(history, indent=2))
    (opt / "train_history.json").write_text(
        (art / "train_history.json").read_text()
    )
    print(
        f"DONE best_loss={history['best_loss']:.6e} @ {history['best_epoch']} "
        f"→ {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
