# Stochastic volatility demos

Two companion exercises on the **same live mid-implied vanilla strip**:

| Demo | Entry point | What it shows |
|------|-------------|---------------|
| **Vanilla surfaces** | `run_vanilla_demo.py` | BS / Heston / 1F / 2F fits; smile, tenor, 3D, residual plots |
| **American barrier** | `run_barrier_demo.py` | Same calibrations → price an American down-and-out call (no closed form) |

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

## 1. Vanilla surface demo

Honest clock counts: Heston / 1F share **one** decay for ATM+skew+smile; 2F
has fast + slow OU clocks. Equal tenor weights. Smile outliers filtered before
fit/plot.

### Vanilla artifacts

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
| `outliers_removed.csv` | Dropped smile outliers |

---

## 2. American barrier demo (no analytical price)

1. Calibrate the four models to live vanillas  
2. Map each fit → risk-neutral dynamics  
3. Price the **same** American down-and-out call with Longstaff–Schwartz MC  
4. Compare prices / knock-out probs / early-exercise premia  

Default contract: \(K\approx 1.05 S_0\), barrier \(B\approx 0.90 S_0\), mid-tenor \(T\).

### Barrier artifacts

| File | Content |
|------|---------|
| `american_barrier_prices.png` | American / European barrier + KO by model |
| `american_vs_european_barrier.png` | Early-exercise premium |
| `american_barrier_vs_2f.png` | Exotic gap vs 2F anchor |
| `american_barrier_vs_B.png` | Price vs barrier |
| `american_barrier_vs_T.png` | Price vs maturity |
| `vanilla_smiles.png` / `vanilla_fit_rmse.png` | Supporting vanilla calibration |

Reports: `fit_report_vanilla.json`, `fit_report_barrier.json` (and
`fit_report.json` mirrors whichever demo ran last).

---

## Shared modules

- `fetch_market.py` — Yahoo chains, parity forwards, mid IVs, outlier filter
- `fit_surfaces.py` — parametric IV models + LS fits
- `american_barrier.py` — path sims + LSMC American DAO call
