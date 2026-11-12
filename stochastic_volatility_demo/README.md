# Stochastic volatility demo

Fit four parametric IV models side by side and show **why two Bergomi
factors** are needed when the surface has a fast short smile **and** a
slow long skew.

1. Black–Scholes (flat σ)
2. Heston-like (**one** clock shared by ATM / skew / smile)
3. 1-factor Bergomi (**one** OU clock)
4. **2-factor Bergomi** (fast + slow OU clocks)

Calibration is honest about clock count: 1F/Heston no longer get free
independent smile decays (that would secretly add extra timescales).
Residuals are equal-weighted by tenor.

## IV extraction (live quotes)

Yahoo’s `impliedVolatility` is **not** used for fitting. Per-expiry forward
\(F\) comes from put-call parity; IVs are re-implied from mid (else last)
with moneyness \(k=\log(K/F)\).

## Run

```bash
pip install -r requirements.txt
python stochastic_volatility_demo/run_demo.py              # synthetic (default; clearest 2F demo)
python stochastic_volatility_demo/run_demo.py synthetic    # same
python stochastic_volatility_demo/run_demo.py ^SPX         # live SPX strip
python stochastic_volatility_demo/run_demo.py auto         # live if 2F advantage is clear, else synthetic
```

Default is a **designed two-timescale** surface (κ_fast≈18, κ_slow≈0.35).
On that strip 2F typically recovers both clocks and beats 1F/Heston by
~10× in RMSE on short *and* long tenors. Live SPX can look flatter; `auto`
falls back to synthetic when the live joint advantage is weak.

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
| `fit_errors_by_moneyness.png` | RMSE by moneyness bucket |
| `fit_report.json` | Params, short/long RMSE, advantage ratios |
