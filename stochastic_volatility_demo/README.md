# Stochastic volatility demo

Download listed equity option quotes, strip a clean implied-volatility (IV)
surface, and **fit** four parametric models side by side:

1. Black–Scholes (flat σ)
2. Heston-like (one CIR clock)
3. 1-factor Bergomi-like (one OU clock)
4. **2-factor Bergomi-like** (fast + slow OU clocks)

The point of the demo is qualitative: only the two-factor surface can match a
steep short-dated smile **and** a milder long-dated skew at the same time.

## IV extraction (important)

Yahoo’s `impliedVolatility` field is **not** used for fitting. For each expiry
the forward \(F\) is implied from near-ATM put-call parity
\(C-P=e^{-r\tau}(F-K)\) (fallback \(S e^{(r-q)\tau}\)). Each quote’s IV is then
re-implied from the mid price (else last) in the Black model with moneyness
\(k=\log(K/F)\). Wide relative spreads and junk last prints are dropped.
Yahoo IV is kept only as a diagnostic (typical MAE vs ours ≈ 1–2%).

## Run

```bash
pip install -r requirements.txt          # includes yfinance
python stochastic_volatility_demo/run_demo.py            # ^SPX (default, densest)
python stochastic_volatility_demo/run_demo.py AAPL       # single-name strip
python stochastic_volatility_demo/run_demo.py auto       # best of AAPL/QQQ/SPY/^SPX
```

Default underlier is **^SPX** — Yahoo’s densest live chain with real bid/ask
(thousands of quotes before a light subsample to ~8×25 for plotting).

**AAPL** (and other single names) also work: after hours Yahoo often blanks
bid/ask, so the fetcher falls back to `lastPrice` for BS inversion. `auto`
mode probes AAPL → QQQ → SPY → ^SPX and keeps the best strip (≥5 tenors),
preferring mid-implied IVs.

Artifacts land in `stochastic_volatility_demo/artifacts/`:

| File | Content |
|------|---------|
| `market_iv_surface.png` | Market IV by forward moneyness / tenor |
| `model_fits_smiles.png` | Fitted smiles vs market (**across moneyness**, fixed τ) |
| `model_fits_tenors.png` | ATM / wing / skew term structures (**across τ**, fixed k) |
| `model_iv_surfaces_3d.png` | 3D IV surfaces: market + all four models + 2F residual |
| `model_surfaces_by_tenor.png` | Per-model smile grids across selected tenors |
| `fit_errors_by_tenor.png` | RMSE by maturity — 2F wins on short+long |
| `fit_errors_by_moneyness.png` | RMSE by log-moneyness bucket |
| `market_quotes.csv` | Filtered quotes used for the fit |
| `market_quotes_live.csv` | Full live snapshot before any synthetic fallback |
| `fit_report.json` | Parameters + RMSE by tenor/moneyness + `hit_bounds` |

If every live download fails, the script falls back to a synthetic
equity-like surface so the demo still runs offline.
