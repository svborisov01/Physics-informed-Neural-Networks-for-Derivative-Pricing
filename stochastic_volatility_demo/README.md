# American barrier demo (no analytical price)

Reformulated around an exotic that **has no closed form**: an
**American down-and-out call**.

## Why this exercise

Vanilla smiles can be fit with parametric IV models. That alone does not
tell you how to price path-dependent early-exercise claims. Here we:

1. **Calibrate** BS / Heston / 1F Bergomi / 2F Bergomi to **live** mid-implied
   vanilla IVs (honest single- vs two-clock structure)
2. **Map** each fit to risk-neutral dynamics
3. **Price** the same American down-and-out call under each dynamics with
   Longstaff–Schwartz Monte Carlo (discrete knock-out + early exercise)
4. **Compare** prices, knock-out probabilities, and early-exercise premia

There is no analytical reference price. Model disagreement *is* the result —
and the 2-factor Bergomi surface (usually the best vanilla fit) is the anchor.

## Contract

Default exotic (from the live spot):

- Call strike \(K \approx 1.05\,S_0\)
- Down-and-out barrier \(B \approx 0.90\,S_0\)
- Maturity ≈ median listed tenor
- American exercise while alive; knocked out if \(S_t \le B\) on the MC grid

## Run

```bash
pip install -r requirements.txt
python stochastic_volatility_demo/run_demo.py           # auto: best live 2F strip
python stochastic_volatility_demo/run_demo.py NVDA
python stochastic_volatility_demo/run_demo.py ^SPX
```

## Artifacts

| File | Content |
|------|---------|
| `american_barrier_prices.png` | American / European barrier prices + KO prob by model |
| `american_vs_european_barrier.png` | Early-exercise premium |
| `american_barrier_vs_2f.png` | Exotic price gap vs 2F anchor |
| `american_barrier_vs_B.png` | Price vs barrier level |
| `american_barrier_vs_T.png` | Price vs maturity |
| `vanilla_smiles.png` | Supporting vanilla calibration |
| `vanilla_fit_rmse.png` | Vanilla IV RMSE by model |
| `fit_report.json` | Fits, dynamics maps, exotic prices |
| `market_quotes.csv` | Cleaned mid-implied vanillas |
| `outliers_removed.csv` | Dropped smile outliers |

## Modules

- `fetch_market.py` — live Yahoo chains, parity forwards, mid IVs, outlier filter
- `fit_surfaces.py` — parametric IV models + equal-tenor LS fits
- `american_barrier.py` — path simulation + LSMC American DAO call
- `run_demo.py` — end-to-end exercise
