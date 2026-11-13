#!/usr/bin/env python3
"""Artifact folder layout for the SV demos.

::

    artifacts/
      common/
        data/     # market quotes, outliers, meta
        logs/
      vanilla/
        data/
        logs/
        graphs/
          comparison/          # multi-model overview plots
          Black-Scholes/
          Heston-like/
          1-factor-Bergomi/
          2-factor-Bergomi/
      barrier/
        data/
        logs/
        graphs/
          comparison/
          <model>/
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART_ROOT = ROOT / "artifacts"

MODEL_DIR_NAMES = {
    "Black–Scholes": "Black-Scholes",
    "Heston-like": "Heston-like",
    "1-factor Bergomi": "1-factor-Bergomi",
    "2-factor Bergomi": "2-factor-Bergomi",
}


def model_dirname(name: str) -> str:
    if name in MODEL_DIR_NAMES:
        return MODEL_DIR_NAMES[name]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def ensure_layout(art_root: Path | None = None) -> dict[str, Path]:
    """Create the standard tree; return useful path handles."""
    root = Path(art_root) if art_root else ART_ROOT
    paths = {
        "root": root,
        "common_data": root / "common" / "data",
        "common_logs": root / "common" / "logs",
        "vanilla_data": root / "vanilla" / "data",
        "vanilla_logs": root / "vanilla" / "logs",
        "vanilla_graphs": root / "vanilla" / "graphs",
        "vanilla_comparison": root / "vanilla" / "graphs" / "comparison",
        "barrier_data": root / "barrier" / "data",
        "barrier_logs": root / "barrier" / "logs",
        "barrier_graphs": root / "barrier" / "graphs",
        "barrier_comparison": root / "barrier" / "graphs" / "comparison",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    for name in MODEL_DIR_NAMES.values():
        (paths["vanilla_graphs"] / name).mkdir(parents=True, exist_ok=True)
        (paths["barrier_graphs"] / name).mkdir(parents=True, exist_ok=True)
    return paths


def model_graph_dir(demo: str, model_name: str, art_root: Path | None = None) -> Path:
    paths = ensure_layout(art_root)
    key = "vanilla_graphs" if demo == "vanilla" else "barrier_graphs"
    d = paths[key] / model_dirname(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def comparison_dir(demo: str, art_root: Path | None = None) -> Path:
    paths = ensure_layout(art_root)
    return paths["vanilla_comparison" if demo == "vanilla" else "barrier_comparison"]
