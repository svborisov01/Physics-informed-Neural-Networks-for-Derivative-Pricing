# Bergomi PINN extensive tests (QMC + ADD)

Benchmark: scrambled Sobol **QMC** with **antithetic differencing (ADD)**.
Checkpoint: `trained_models/bergomi.pt`.

## Where to find plots

### Error heatmaps (τ, S)
`smile_heatmaps/heatmap_*.png` — one per smile set; title states \((ξ₀,ω,κ,ρ,X)\).

### 3D surfaces (price + key Greeks)
`surfaces_3d/<smile>/3d_*.png` (interactive HTML alongside):

| File | Quantity |
|------|----------|
| `3d_price_*.png` | PINN price \(V\) |
| `3d_delta_*.png` | Delta \(\partial V/\partial S\) |
| `3d_gamma_*.png` | Gamma \(\partial^2 V/\partial S^2\) |
| `3d_theta_*.png` | Theta \(\partial V/\partial t\) |
| `3d_vega_*.png` | Vega \(\partial V/\partial\sigma\), \(\sigma=\sqrt{ξ₀}\) |
| `3d_dual_X_*.png` | Factor greek \(\partial V/\partial X\) |
| `3d_abs_err_*.png` | 3D abs price error vs QMC+ADD |
| `3d_rel_err_*.png` | 3D relative price error |

## Fine-grid abs norms (baseline)

| L1 | L2 | L∞ |
|----|----|-----|
| 0.106 | 0.135 | 0.321 |

## Smile heatmap abs norms

| Smile | Abs L1 | Abs L∞ |
|-------|--------|--------|
| baseline | 0.117 | 0.297 |
| strong_vovol | 0.191 | 0.507 |
| strong_neg_skew | 0.228 | 0.587 |
| mild_fast_mr | 0.053 | 0.205 |
| slow_mr | 0.287 | 0.919 |
| near_bs | 0.005 | 0.025 |

## Reproduce plots only (fast)

```bash
python scripts/plot_bergomi_surfaces.py --checkpoint trained_models/bergomi.pt
```

Full QMC+ADD suite:

```bash
python scripts/run_bergomi_extensive_tests.py --checkpoint trained_models/bergomi.pt
```
