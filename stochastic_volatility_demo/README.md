# Stochastic volatility demo

Download listed equity option quotes, strip a clean implied-volatility (IV)
surface, and **fit** four parametric models side by side:

1. Black–Scholes (flat σ)
2. Heston-like (one CIR clock)
3. 1-factor Bergomi-like (one OU clock, lognormal wings)
4. **2-factor Bergomi-like** (fast + slow OU clocks)

The point of the demo is qualitative: only the two-factor surface can match a
steep short-dated smile **and** a milder long-dated skew at the same time.

## Run

```bash
pip install -r requirements.txt          # includes yfinance
python stochastic_volatility_demo/run_demo.py
```

Artifacts land in `stochastic_volatility_demo/artifacts/`:

| File | Content |
|------|---------|
| `market_iv_surface.png` | Market IV by moneyness / tenor |
| `model_fits_smiles.png` | Fitted smiles vs market at several tenors |
| `fit_errors_by_tenor.png` | RMSE by maturity — 2F wins on short+long |
| `market_quotes.csv` | Filtered quotes used for the fit |
| `fit_report.json` | Parameters + RMSE table |

Default underlier: **SPY** via Yahoo Finance (`yfinance`). If the live
download fails (weekend / rate limit), the script falls back to a cached
CSV or a realistic synthetic surface so the demo still runs offline.
