# Stochastic volatility demos

Two companion exercises on the **same live mid-implied vanilla strip**.
All outputs land in one place: [`results/sv_demo/`](../results/sv_demo/).

| Demo | Entry point | What it shows |
|------|-------------|---------------|
| **Vanilla** | `run_vanilla_demo.py` | BS / Heston / 1F / 2F fits + Greeks |
| **Barrier** | `run_barrier_demo.py` | American down-and-out call + KO Greeks |

```bash
pip install -r requirements.txt

python -m stochastic_volatility_demo.run_demo              # both
python -m stochastic_volatility_demo.run_vanilla_demo NVDA
python -m stochastic_volatility_demo.run_barrier_demo NVDA
```

---

## Output layout

```
results/sv_demo/
  market_quotes.csv
  market_meta.json
  outliers_removed.csv
  vanilla/
    fit_report.json
    model_fits_smiles.png …
    greeks_delta.png …          # comparison across models
    Black-Scholes/greeks.png
    Heston-like/greeks.png
    1-factor-Bergomi/greeks.png
    2-factor-Bergomi/greeks.png
  barrier/
    fit_report.json
    american_barrier_*.png …
    greeks_*.png
    <model>/greeks.png
```

PINN training dumps stay under `results/{bergomi,heston,…}/` — separate from this demo.

---

## Greeks

Comparable units for every model:

| Greek | Unit |
|-------|------|
| Delta / Gamma | ∂V/∂S, ∂²V/∂S² |
| Theta | per calendar day |
| Vega | per +1 percentage point of spot vol |

- **Vanilla BS**: closed form.
- **Vanilla SV**: finite difference on European MC with common random numbers.
- **Barrier BS**: Haug European down-and-out closed form (+ FD).
- **Barrier SV**: European KO Monte Carlo (American EEP is tiny; LSMC noise otherwise dominates Gamma/Theta). American LSMC prices are still the headline exotic numbers.

---

## Modules

- `fetch_market.py` — Yahoo chains, parity forwards, mid IVs, outlier filter
- `fit_surfaces.py` — parametric IV models + LS fits
- `american_barrier.py` — path sims, LSMC American DAO, KO Greeks
- `greeks_plots.py` — panels into `results/sv_demo/{vanilla,barrier}/`
- `output_layout.py` — single output tree helper
