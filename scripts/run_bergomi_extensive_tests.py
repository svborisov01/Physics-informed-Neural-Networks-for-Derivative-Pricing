#!/usr/bin/env python3
"""CLI entry point for extensive Bergomi PINN vs QMC+ADD tests."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from support_tools.bergomi_test_suite import run_extensive_bergomi_tests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="trained_models/bergomi.pt",
        help="Path to Bergomi PINN checkpoint",
    )
    p.add_argument(
        "--out-dir",
        default="artifacts/bergomi_extensive_tests",
        help="Directory for heatmaps, npz, and report.json",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-paths-price", type=int, default=131_072)
    p.add_argument("--n-paths-greeks", type=int, default=262_144)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument(
        "--copy-artifacts",
        default="/opt/cursor/artifacts/bergomi_extensive_tests",
        help="Optional mirror directory for Cursor artifacts",
    )
    args = p.parse_args()

    report = run_extensive_bergomi_tests(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device=args.device,
        n_paths_price=args.n_paths_price,
        n_paths_greeks=args.n_paths_greeks,
        n_steps=args.n_steps,
    )

    if args.copy_artifacts:
        dest = Path(args.copy_artifacts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(args.out_dir, dest)
        print(f"Mirrored artifacts to {dest}")

    fine = report["fine_grid"]["abs_norms"]
    print(
        "\nSummary — fine-grid abs norms: "
        f"L1={fine['L1']:.4f}, L2={fine['L2']:.4f}, Linf={fine['Linf']:.4f}"
    )
    for g in report["greeks"]:
        name = g["smile"]["name"]
        print(
            f"Greeks[{name}] Delta L1={g['delta_norms']['L1']:.4f} "
            f"Theta L1={g['theta_norms']['L1']:.4f}"
        )


if __name__ == "__main__":
    main()
