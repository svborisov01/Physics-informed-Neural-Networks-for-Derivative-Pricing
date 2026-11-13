#!/usr/bin/env python3
"""
Stochastic-volatility demos (two exercises, same live market strip):

  vanilla  — fit BS / Heston / 1F / 2F to listed vanillas; smile/tenor/surface plots
  barrier  — price an American down-and-out call under each calibrated dynamics
             (no closed form; Longstaff–Schwartz MC)

Usage
-----
  python -m stochastic_volatility_demo.run_demo              # both (default)
  python -m stochastic_volatility_demo.run_demo all NVDA
  python -m stochastic_volatility_demo.run_demo vanilla NVDA
  python -m stochastic_volatility_demo.run_demo barrier NVDA

Or call the modules directly:

  python -m stochastic_volatility_demo.run_vanilla_demo NVDA
  python -m stochastic_volatility_demo.run_barrier_demo NVDA
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = "all"
    symbol = "auto"
    if argv:
        if argv[0].lower() in {"all", "vanilla", "barrier", "exotic", "american"}:
            mode = argv[0].lower()
            symbol = argv[1] if len(argv) > 1 else "auto"
        else:
            # `run_demo.py NVDA` → both demos on NVDA
            symbol = argv[0]

    if mode in {"exotic", "american"}:
        mode = "barrier"

    if mode in {"all", "vanilla"}:
        print("=" * 64)
        print("VANILLA SURFACE DEMO")
        print("=" * 64)
        from stochastic_volatility_demo.run_vanilla_demo import main as vanilla_main

        vanilla_main(symbol)

    if mode in {"all", "barrier"}:
        print("\n" + "=" * 64)
        print("AMERICAN DOWN-AND-OUT CALL DEMO (no closed form)")
        print("=" * 64)
        from stochastic_volatility_demo.run_barrier_demo import main as barrier_main

        barrier_main(symbol)


if __name__ == "__main__":
    main()
