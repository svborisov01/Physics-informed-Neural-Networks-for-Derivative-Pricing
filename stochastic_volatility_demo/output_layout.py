#!/usr/bin/env python3
"""Single output tree for the SV demos (under ``results/sv_demo/``).

::

    results/sv_demo/
      market_quotes.csv
      market_meta.json
      outliers_removed.csv
      vanilla/
        fit_report.json
        <comparison plots>.png
        Black-Scholes/greeks.png
        Heston-like/greeks.png
        ...
      barrier/
        fit_report.json
        <comparison plots>.png
        <model>/greeks.png
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "results" / "sv_demo"

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


def ensure_layout(out_root: Path | None = None) -> dict[str, Path]:
    root = Path(out_root) if out_root else OUT_ROOT
    paths = {
        "root": root,
        "vanilla": root / "vanilla",
        "barrier": root / "barrier",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    for name in MODEL_DIR_NAMES.values():
        (paths["vanilla"] / name).mkdir(parents=True, exist_ok=True)
        (paths["barrier"] / name).mkdir(parents=True, exist_ok=True)
    return paths


def model_dir(demo: str, model_name: str, out_root: Path | None = None) -> Path:
    paths = ensure_layout(out_root)
    d = paths[demo] / model_dirname(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def demo_dir(demo: str, out_root: Path | None = None) -> Path:
    return ensure_layout(out_root)[demo]


def clear_demo_outputs(out_root: Path | None = None) -> Path:
    """Remove previous demo plots/reports so the tree stays flat and current."""
    root = Path(out_root) if out_root else OUT_ROOT
    if root.exists():
        shutil.rmtree(root)
    return ensure_layout(root)["root"]
