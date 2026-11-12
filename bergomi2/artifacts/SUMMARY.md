# 2-Factor Bergomi PINN — results

## Training

| Item | Value |
|------|-------|
| Checkpoint | `trained_models/bergomi2.pt` |
| Architecture | hidden=96, depth=4, **28 features** |
| Positivity | softplus β=400 |
| Baseline | BS with `spot_var` σ_eff |
| Collocation | N_pde=1500, N_boundary=600 |
| Epochs | 1500 total (800 + 700 continue) |
| Best train loss | ≈ **1.07×10⁻³** (epoch 370 of continue phase) |
| Wall time | ≈ 90 s CPU |

Optimal feature list: see `pricing/bergomi2_option_pricing.py` → `FEATURE_NAMES`
and `bergomi2/README.md`.

## Accuracy vs Monte Carlo

Market: \(\xi_0=0.04\), \((\omega_1,\omega_2)=(1.0,0.6)\),
\((\kappa_1,\kappa_2)=(4.0,0.8)\), \((\rho_1,\rho_2,\rho_{12})=(-0.6,-0.3,0.2)\),
\(r=0.05\), \(X_1=X_2=0\), \(K=100\).

Grid: \(S\in\{80,\ldots,130\}\), \(\tau\in[0.25,1.5]\) (6 mats), MC 40k paths.

| Metric | Value |
|--------|-------|
| RMSE | ≈ 0.34 |
| MAE | ≈ 0.30 |
| Max \|err\| | ≈ 0.58 |
| ATM \(T=1\) PINN | 10.00 |
| ATM \(T=1\) MC | 10.11 ± 0.04 |

Normalized (\(u=V/K\)): RMSE ≈ 0.0034 — usable first checkpoint; longer training /
more paths will tighten further.

## Artifacts

- `training_loss.png` — PDE / terminal / boundary losses
- `abs_err_heatmap.png` — \|PINN − MC\| on \((S,\tau)\)
- `atm_term_structure.png` — ATM prices vs maturity
- `test_report.json` — numeric report
- `vol_smile_richness.png` — stylized smile comparison BS / Heston / 1F / 2F
- `vol_termstructure_richness.png` — ATM & skew term structures
- `vol_surface_bs_vs_bergomi2.png` — 3D IV: flat vs two-factor

## Volatility-surface richness (short explanation)

1. **Black–Scholes** — one number \(\sigma\): surface is flat in strike and maturity.
2. **Heston** — one CIR variance factor: generates skew/smile, but skew dies with
   a **single** mean-reversion clock.
3. **1-factor Bergomi** — one OU factor on log forward variance: similar
   one-timescale richness; often stronger short wings (lognormal variance).
4. **2-factor Bergomi** — fast + slow OU factors: short skew controlled by
   \((\omega_1,\kappa_1,\rho_1)\), long skew / ATM slope by
   \((\omega_2,\kappa_2,\rho_2)\) → much richer term structure of the IV surface.
