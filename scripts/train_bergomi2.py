#!/usr/bin/env python3
"""Train the 2-factor Bergomi PINN and save checkpoint + history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pricing.bergomi2_option_pricing import FEATURE_NAMES, PINN, train_network


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-pde", type=int, default=3500)
    p.add_argument("--n-boundary", type=int, default=1200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="results/bergomi2")
    p.add_argument("--checkpoint", default="checkpoints/bergomi2.pt")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    pinn = PINN(
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
        hidden=args.hidden,
        depth=args.depth,
        v_max=1.0,
        kappa_floor=0.25,
    ).to(device)

    print(f"Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    print(f"Params: {sum(p.numel() for p in pinn.parameters()):,}")

    history = train_network(
        pinn,
        N_pde=args.n_pde,
        N_boundary=args.n_boundary,
        epochs=args.epochs,
        lr=args.lr,
        sigma_mode="flat_fwd",
        stationary=True,
        detach_source=False,
        relative_bs=True,
        balance_losses=False,
        use_mc_anchor=True,
        lambda_data=2.0,
        print_every=args.print_every,
        best_model_path=args.checkpoint,
        save_model=True,
    )

    # Save history JSON (lists only)
    hist_path = out_dir / "training_history.json"
    serializable = {
        k: (v if not isinstance(v, (np.ndarray,)) else v.tolist())
        for k, v in history.items()
        if k != "feature_names"
    }
    serializable["feature_names"] = FEATURE_NAMES
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    ax.semilogy(history["epoch"], history["total"], label="total")
    ax.semilogy(history["epoch"], history["physics"], label="PDE", alpha=0.8)
    ax.semilogy(history["epoch"], history["terminal"], label="terminal", alpha=0.8)
    ax.semilogy(history["epoch"], history["boundary"], label="boundary", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("2-factor Bergomi PINN training")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Best loss {history['best_loss']:.6e} @ epoch {history['best_epoch']}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    main()
