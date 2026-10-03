# Bergomi PINN extensive tests (QMC + ADD)

Benchmark: scrambled Sobol **QMC** with **antithetic differencing (ADD)**.
Checkpoint: `trained_models/bergomi.pt`.
Positivity: softplus β=400.0 (negatives on fine grid: 0/1479, min V=0.09612).

## Where to find plots

### Error heatmaps (τ, S)
`smile_heatmaps/heatmap_*.png` — one per smile set; title states \((ξ₀,ω,κ,ρ,X)\).

### 3D surfaces (price + key Greeks)
`surfaces_3d/<smile>/3d_*.png` (interactive HTML alongside):

| File | Quantity |
|------|----------|
| `3d_price_*.png` | PINN price $V$ |
| `3d_delta_*.png` | Delta $\partial V/\partial S$ |
| `3d_gamma_*.png` | Gamma $\partial^2 V/\partial S^2$ |
| `3d_theta_*.png` | Theta $\partial V/\partial t$ |
| `3d_vega_*.png` | Vega $\partial V/\partial\sigma$, $\sigma=\sqrt{ξ₀}$ |
| `3d_dual_X_*.png` | Factor greek $\partial V/\partial X$ |
| `3d_abs_err_*.png` | 3D abs price error vs QMC+ADD |
| `3d_rel_err_*.png` | 3D relative price error |

## Fine-grid abs norms (baseline)

| L1 | L2 | L∞ | neg prices | min V |
|----|----|-----|------------|-------|
| 0.110 | 0.137 | 0.321 | 0 | 0.09612 |

## Smile heatmap abs norms

| Smile | Abs L1 | Abs L∞ | neg | min V |
|-------|--------|--------|-----|-------|
| baseline | 0.127 | 0.297 | 0 | 0.1382 |
| strong_vovol | 0.195 | 0.507 | 0 | 0.0865 |
| strong_neg_skew | 0.237 | 0.587 | 0 | 0.1548 |
| mild_fast_mr | 0.061 | 0.205 | 0 | 0.1855 |
| slow_mr | 0.286 | 0.919 | 0 | 0.08561 |
| near_bs | 0.014 | 0.172 | 0 | 0.1722 |

## Reproduce plots only (fast)

```bash
python scripts/plot_bergomi_surfaces.py --checkpoint trained_models/bergomi.pt
```

Full QMC+ADD suite:

```bash
python scripts/run_bergomi_extensive_tests.py --checkpoint trained_models/bergomi.pt
```
