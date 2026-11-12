# 2-Factor Bergomi PINN — results

## Training

| Item | Value |
|------|-------|
| Checkpoint | `checkpoints/bergomi2.pt` |
| Architecture | hidden=96, depth=4, **28 features** |
| Positivity | softplus β=400 |
| Baseline | BS with **`flat_fwd`** σ_eff |
| PDE loss | relative to \|L[u_BS]\| + Huber; λ_pde anneal |
| Extras | **MC price anchors** (36-point grid), short-τ collocation bias |
| Epochs | 1000 |
| Best selection loss | ≈ **4.24×10⁻⁴** (epoch 919) |

## Accuracy vs Monte Carlo

Market: \(\xi_0=0.04\), \((\omega_1,\omega_2)=(1.0,0.6)\),
\((\kappa_1,\kappa_2)=(4.0,0.8)\), \((\rho_1,\rho_2,\rho_{12})=(-0.6,-0.3,0.2)\),
\(r=0.05\), \(X_1=X_2=0\), \(K=100\).

Grid: \(S\in\{80,\ldots,130\}\), \(\tau\in[0.25,1.5]\) (6 mats), MC 40k paths.

| Metric | Prior (PDE-only) | This run |
|--------|------------------|----------|
| RMSE | ≈ 0.39 | ≈ **0.31** |
| MAE | ≈ 0.32 | ≈ **0.25** |
| Max \|err\| | ≈ 0.66 | ≈ **0.59** |
| ATM \(T=1\) PINN | 9.88 | **10.29** |
| ATM \(T=1\) MC | 10.11 ± 0.04 | 10.11 ± 0.04 |

## Artifacts

Training curves, error heatmaps, stylized IV smiles, and
[`SURFACE_COMPARISON.md`](SURFACE_COMPARISON.md).
