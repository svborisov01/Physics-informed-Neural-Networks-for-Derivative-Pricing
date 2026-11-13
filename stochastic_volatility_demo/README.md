# Stochastic volatility demos

Two companion exercises on the **same live mid-implied vanilla strip**:

| Demo | Entry point | What it shows |
|------|-------------|---------------|
| **Vanilla surfaces** | `run_vanilla_demo.py` | BS / Heston / 1F / 2F fits; smile, tenor, 3D, residual, Greeks plots |
| **American barrier** | `run_barrier_demo.py` | Same calibrations → American down-and-out call + Greeks (no closed form) |

```bash
pip install -r requirements.txt

# both (default)
python -m stochastic_volatility_demo.run_demo
python -m stochastic_volatility_demo.run_demo all NVDA

# individually
python -m stochastic_volatility_demo.run_vanilla_demo NVDA
python -m stochastic_volatility_demo.run_barrier_demo NVDA
```

`auto` probes NVDA, QQQ, AAPL, …, ^SPX and keeps the live strip with the
strongest joint short+long 2F vanilla advantage.

---

## Artifact layout

```
artifacts/
  common/
    data/          # market_quotes.csv, outliers_removed.csv, market_meta.json
    logs/
  vanilla/
    data/          # fit_report_vanilla.json
    logs/          # run_summary.txt
    graphs/
      comparison/  # multi-model overview + greeks_*.png
      Black-Scholes/
      Heston-like/
      1-factor-Bergomi/
      2-factor-Bergomi/   # per-model greeks.png, delta.png, …
  barrier/
    data/          # fit_report_barrier.json
    logs/
    graphs/
      comparison/
      <model>/
```

Flat copies of key reports / CSVs remain at `artifacts/` for backward
compatibility. Flat PNG paths from older runs are superseded by the tree above.

---

## 1. Vanilla surface demo

Honest clock counts: Heston / 1F share **one** decay for ATM+skew+smile; 2F
has fast + slow OU clocks. Equal tenor weights. Smile outliers filtered before
fit/plot.

### Comparison graphs (`vanilla/graphs/comparison/`)

| File | Content |
|------|---------|
| `model_fits_smiles.png` | Smiles across moneyness |
| `model_fits_tenors.png` | ATM / wing / skew term structures |
| `model_iv_surfaces_3d.png` | 3D surfaces for all four models |
| `model_surfaces_by_tenor.png` | Per-model smile grids |
| `model_short_long_contrast.png` | Short vs long smile proof |
| `fit_short_vs_long.png` | Short/long RMSE bars |
| `model_residuals.png` | Residual heatmaps |
| `fit_errors_by_tenor.png` / `fit_errors_by_moneyness.png` | Error breakdowns |
| `outliers_vs_kept.png` | Kept vs removed outliers |
| `greeks_delta.png` … `greeks_vega.png` | European call Greeks by model |

### Per-model graphs (`vanilla/graphs/<model>/`)

`greeks.png` (Δ/Γ/Θ/Vega panel) plus `delta.png`, `gamma.png`, `theta.png`,
`vega.png`. BS uses closed form; SV models use FD on European MC.

---

## 2. American barrier demo (no analytical price)

1. Calibrate the four models to live vanillas  
2. Map each fit → risk-neutral dynamics  
3. Price the **same** American down-and-out call with Longstaff–Schwartz MC  
4. FD Greeks (Δ, Γ, Θ, Vega) under each dynamics  
5. Compare prices / knock-out probs / early-exercise premia / Greeks  

Default contract: \(K\approx 1.05 S_0\), barrier \(B\approx 0.90 S_0\), mid-tenor \(T\).

### Comparison graphs (`barrier/graphs/comparison/`)

| File | Content |
|------|---------|
| `american_barrier_prices.png` | American / European barrier + KO by model |
| `american_vs_european_barrier.png` | Early-exercise premium |
| `american_barrier_vs_2f.png` | Exotic gap vs 2F anchor |
| `american_barrier_vs_B.png` | Price vs barrier |
| `american_barrier_vs_T.png` | Price vs maturity |
| `vanilla_smiles.png` / `vanilla_fit_rmse.png` | Supporting vanilla calibration |
| `greeks_delta.png` … `greeks_vega.png` | American DAO Greeks by model |

### Per-model graphs (`barrier/graphs/<model>/`)

Same Greek panel / individual files as vanilla, for the American DAO call.

Reports: `fit_report_vanilla.json`, `fit_report_barrier.json` (and
`fit_report.json` mirrors whichever demo ran last).

---

## Shared modules

- `fetch_market.py` — Yahoo chains, parity forwards, mid IVs, outlier filter
- `fit_surfaces.py` — parametric IV models + LS fits
- `american_barrier.py` — path sims + LSMC American DAO + FD Greeks
- `greeks_plots.py` — vanilla / barrier Greek panels into model folders
- `artifacts_layout.py` — `common` / `vanilla` / `barrier` × `data` / `logs` / `graphs`
