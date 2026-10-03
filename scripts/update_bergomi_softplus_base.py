#!/usr/bin/env python3
"""
Promote main's bergomi.pt weights to the softplus-β=400 base checkpoint.

Keeps the same network weights (no PDE retrain); writes
``price_softplus_beta=400`` into checkpoint metadata so load_model enables
softplus positivity by default.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pricing.bergomi_option_pricing import DEFAULT_PRICE_SOFTPLUS_BETA
from support_tools.model_wrapper import load_model, predict_price


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="trained_models/bergomi.pt")
    p.add_argument("--dst", default="trained_models/bergomi.pt")
    p.add_argument(
        "--backup",
        default="trained_models/bergomi_pre_softplus.pt",
        help="Copy of pre-softplus weights (empty string to skip)",
    )
    p.add_argument("--beta", type=float, default=DEFAULT_PRICE_SOFTPLUS_BETA)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_file():
        raise SystemExit(f"missing checkpoint: {src}")

    if args.backup:
        backup = Path(args.backup)
        if not backup.is_file() or backup.resolve() != src.resolve():
            shutil.copy2(src, backup)
            print(f"Backup → {backup}")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    ckpt["price_softplus_beta"] = float(args.beta)
    ckpt["positivity"] = "softplus"
    ckpt["softplus_mode"] = "inference_wrap_base"
    ckpt["softplus_note"] = (
        "Base Bergomi model: same PDE-trained weights as pre-softplus "
        f"bergomi.pt, with softplus_β(u_BS+U), β={args.beta}."
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print(f"Wrote {dst} with price_softplus_beta={args.beta}")

    pinn, meta = load_model(str(dst), device="cpu")
    pinn.eval()
    assert float(meta.get("price_softplus_beta", -1)) == float(args.beta)
    assert float(pinn.price_softplus_beta) == float(args.beta)

    # Deep OTM should be non-negative under softplus.
    v_otm = float(
        predict_price(
            pinn,
            S=70.0,
            K=100.0,
            tau=0.4,
            X=0.0,
            r=0.05,
            xi0=0.04,
            omega=1.0,
            kappa=2.0,
            rho=-0.5,
        )
        .detach()
        .cpu()
    )
    v_atm = float(
        predict_price(
            pinn,
            S=100.0,
            K=100.0,
            tau=1.0,
            X=0.0,
            r=0.05,
            xi0=0.04,
            omega=1.0,
            kappa=2.0,
            rho=-0.5,
        )
        .detach()
        .cpu()
    )
    print(f"Smoke OTM V={v_otm:.6f} (expect >= 0) | ATM V={v_atm:.6f}")
    if v_otm < 0:
        raise SystemExit("softplus base still produced a negative OTM price")
    print("OK")


if __name__ == "__main__":
    main()
