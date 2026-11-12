# Stochastic volatility demo

Fit four parametric IV models on **real listed option quotes** and show why
two Bergomi factors help when the surface has a fast short smile **and** a
slow long skew.

1. Black–Scholes (flat σ)
2. Heston-like (**one** clock shared by ATM / skew / smile)
3. 1-factor Bergomi (**one** OU clock)
4. **2-factor Bergomi** (fast + slow OU clocks)

Calibration is honest about clock count: 1F/Heston share a single decay for
ATM+skew+smile. Residuals are equal-weighted by tenor.

## IV extraction

Yahoo’s `impliedVolatility` is **not** used for fitting. Per-expiry forward
\(F\) comes from put-call parity; IVs are re-implied from mid (else last)
with moneyness \(k=\log(K/F)\). Wide spreads are dropped. Per-tenor **smile
outliers** are removed via a robust quadratic fit (MAD z-score, absolute
residual, and neighbor-jump spikes) before calibration.

## Run

```bash
pip install -r requirements.txt
python stochastic_volatility_demo/run_demo.py           # auto: best live 2F strip
python stochastic_volatility_demo/run_demo.py NVDA      # pin a symbol
python stochastic_volatility_demo/run_demo.py QQQ
python stochastic_volatility_demo/run_demo.py ^SPX
```

Default `auto` probes NVDA, QQQ, AAPL, MSFT, …, ^SPX and keeps the live
strip with the strongest joint short+long 2F advantage (all mid-implied).
Synthetic is **not** used unless you pass `synthetic` explicitly.

Artifacts in `stochastic_volatility_demo/artifacts/`:

| File | Content |
|------|---------|
| `model_short_long_contrast.png` | Short vs long smile — headline 2F proof |
| `fit_short_vs_long.png` | Short-tenor vs long-tenor RMSE bars |
| `model_residuals.png` | Residual heatmaps (Heston / 1F / 2F) |
| `model_fits_smiles.png` | Smiles across moneyness |
| `model_fits_tenors.png` | ATM / wing / skew term structures |
| `model_iv_surfaces_3d.png` | 3D surfaces for all four models |
| `model_surfaces_by_tenor.png` | Per-model smile grids |
| `fit_errors_by_tenor.png` | RMSE by maturity |
| `fit_report.json` | Params, short/long RMSE, candidate probes |
| `outliers_removed.csv` | Smile outliers dropped before calibration |
| `live_probe.json` | Optional full multi-symbol probe log |
