#!/usr/bin/env python3
"""
Generate Bergomi error heatmaps (if missing) and 3D surfaces for price + Greeks.

Uses existing QMC+ADD npz grids when available; only the PINN autograd grids
are recomputed (fast).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from support_tools.bergomi_test_suite import (
    DEFAULT_SMILE_SETS,
    run_3d_error_surfaces,
    run_3d_surfaces,
    save_heatmap,
)
from support_tools.model_wrapper import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="trained_models/bergomi.pt")
    p.add_argument("--out-dir", default="artifacts/bergomi_extensive_tests")
    p.add_argument(
        "--copy-artifacts",
        default="/opt/cursor/artifacts/bergomi_extensive_tests",
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pinn, meta = load_model(args.checkpoint, device=args.device)
    pinn.eval()
    print(f"Loaded {args.checkpoint} | sigma_mode={meta.get('sigma_mode')}")

    surface_index = []
    for smile in DEFAULT_SMILE_SETS:
        print(f"\n=== {smile.title_fragment()} ===")
        smile_dir = out_dir / "surfaces_3d" / smile.name
        surf = run_3d_surfaces(pinn, smile, out_dir=smile_dir, write_html=True)
        print("  wrote price/Delta/Gamma/Theta/Vega/dual-X 3D PNGs + HTML")

        npz = out_dir / "smile_heatmaps" / f"fine_grid_{smile.name}.npz"
        # also ensure heatmap PNGs exist from npz
        if npz.exists():
            data = __import__("numpy").load(npz)
            heat_dir = out_dir / "smile_heatmaps"
            save_heatmap(
                data["S_grid"],
                data["tau_grid"],
                data["abs_err"],
                title=(
                    f"Price-error norm |PINN − QMC+ADD| on (τ, S)\n"
                    f"Smile params — {smile.title_fragment()}"
                ),
                path=heat_dir / f"heatmap_{smile.name}.png",
                cbar_label="|ΔV| (price norm)",
            )
            err3d = run_3d_error_surfaces(npz, smile, out_dir=smile_dir, write_html=True)
            surf["error_3d"] = err3d
            print("  refreshed error heatmap + 3D error surfaces")
        else:
            print(f"  warning: missing {npz}; skip error surfaces")

        surface_index.append(surf)

    index_path = out_dir / "surfaces_3d" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(surface_index, f, indent=2)
    print(f"\nWrote {index_path}")

    if args.copy_artifacts:
        dest = Path(args.copy_artifacts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # mirror whole out_dir so heatmaps + surfaces are visible
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out_dir, dest)
        print(f"Mirrored artifacts to {dest}")


if __name__ == "__main__":
    main()
