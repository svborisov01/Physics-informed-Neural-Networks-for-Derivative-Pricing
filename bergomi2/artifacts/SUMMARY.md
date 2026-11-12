# 2-Factor Bergomi PINN — results

## Training

| Item | Value |
|------|-------|
| Checkpoint | `trained_models/bergomi2.pt` |
| Architecture | hidden=96, depth=4, **28 features** |
| Positivity | softplus β=400 |
| Baseline | BS with `spot_var` σ_eff |
| PDE loss | **relative to \|L[u_BS]\|** + Huber; λ_pde anneal 0.5→1.0 |
| Correction gate | corr_scale ≈ 0.18 (mildly wider than 1F) |
| Collocation | N_pde=2500, N_boundary=1000 |
| Epochs | 1200 |
| Best selection loss | ≈ **4.82×10⁻⁴** (epoch 964) |
| Min relative PDE | ≈ **4.0×10⁻⁴** |
| Absolute-scaled PDE (post-hoc) | ≈ **8.4×10⁻⁴** (was ≈1.4×10⁻³ on v1) |
| Wall time | ≈ 98 s CPU |

Relative PDE scaling removes the absolute \|L[u_BS]\| floor under Bergomi
dynamics so the physics term can keep decreasing; checkpoint selection
weights terminal/boundary losses so price accuracy is preserved.

Optimal feature list: see `pricing/bergomi2_option_pricing.py` → `FEATURE_NAMES`
and `bergomi2/README.md`.

## Accuracy vs Monte Carlo

Market: \(\xi_0=0.04\), \((\omega_1,\omega_2)=(1.0,0.6)\),
\((\kappa_1,\kappa_2)=(4.0,0.8)\), \((\rho_1,\rho_2,\rho_{12})=(-0.6,-0.3,0.2)\),
\(r=0.05\), \(X_1=X_2=0\), \(K=100\).

Grid: \(S\in\{80,\ldots,130\}\), \(\tau\in[0.25,1.5]\) (6 mats), MC 40k paths.

| Metric | Value |
|--------|-------|
| RMSE | ≈ 0.39 |
| MAE | ≈ 0.32 |
| Max \|err\| | ≈ 0.66 |
| ATM \(T=1\) PINN | 9.88 |
| ATM \(T=1\) MC | 10.11 ± 0.04 |

Normalized (\(u=V/K\)): RMSE ≈ 0.0039.

## Artifacts

- `training_loss.png` — PDE / terminal / boundary losses
- `abs_err_heatmap.png` — \|PINN − MC\| on \((S,\tau)\)
- `atm_term_structure.png` — ATM prices vs maturity
- `test_report.json` — numeric report
- `vol_smile_richness.png` — stylized smile comparison BS / Heston / 1F / 2F
- `vol_termstructure_richness.png` — ATM & skew term structures
- `vol_surface_bs_vs_bergomi2.png` — 3D IV: flat BS vs **U-shaped** 2F smile
- `vol_smile_bergomi2_short.png` — short-dated smile slices vs flat BS

## Volatility-surface richness

Full prose: see [`../SURFACE_COMPARISON.md`](../SURFACE_COMPARISON.md).

1. **Black–Scholes** — one number \(\sigma\): surface is flat in strike and maturity.
2. **Heston** — one CIR variance factor: generates skew/smile, but skew dies with
   a **single** mean-reversion clock.
3. **1-factor Bergomi** — one OU factor on log forward variance: similar
   one-timescale richness; often stronger short wings (lognormal variance).
4. **2-factor Bergomi** — fast + slow OU factors: short skew controlled by
   \((\omega_1,\kappa_1,\rho_1)\), long skew / ATM slope by
   \((\omega_2,\kappa_2,\rho_2)\) → much richer term structure of the IV surface.
