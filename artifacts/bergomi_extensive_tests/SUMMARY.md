# Bergomi PINN extensive tests (QMC + ADD)

Benchmark: scrambled Sobol **QMC** with **antithetic differencing (ADD)**.
Checkpoint: `trained_models/bergomi.pt`. Paths: 131072 (prices), 131072 (Greeks).

## 1) Fine-grid price norms (baseline smile)

Grid: \(S\in[60,160]\) step 2, \(\tau\in[0.2,1.6]\) (29 points).

| Norm | Absolute \|PINN−QMC+ADD\| |
|------|---------------------------|
| L1 / MAE | 0.106 |
| L2 / RMSE | 0.135 |
| L∞ | 0.321 |

## 2) Smile-structure heatmaps

| Smile | (ξ₀, ω, κ, ρ) | Abs L1 | Abs L2 | Abs L∞ |
|-------|---------------|--------|--------|--------|
| baseline | (0.04, 1.0, 2.0, −0.5) | 0.117 | 0.144 | 0.297 |
| strong_vovol | (0.04, 1.8, 2.0, −0.5) | 0.191 | 0.226 | 0.507 |
| strong_neg_skew | (0.04, 1.0, 2.0, −0.85) | 0.228 | 0.278 | 0.587 |
| mild_fast_mr | (0.04, 0.5, 4.0, −0.3) | 0.053 | 0.064 | 0.205 |
| slow_mr | (0.04, 1.2, 0.6, −0.6) | 0.287 | 0.361 | 0.919 |
| near_bs | (0.04, 0.05, 2.0, 0.0) | 0.005 | 0.007 | 0.025 |

Heatmaps: `smile_heatmaps/heatmap_*.png` (axes τ, S; color = abs price error; title states smile params).

## 3) Greeks PINN vs QMC+ADD

Pathwise Delta + CRN finite-difference Theta.

| Smile | Delta L1 | Delta L∞ | Theta L1 | Theta L∞ |
|-------|----------|----------|----------|----------|
| baseline | 0.011 | 0.027 | 0.257 | 0.554 |
| strong_vovol | 0.018 | 0.050 | 0.415 | 0.987 |
| strong_neg_skew | 0.019 | 0.045 | 0.320 | 0.834 |

## Reproduce

```bash
python scripts/run_bergomi_extensive_tests.py \
  --checkpoint trained_models/bergomi.pt \
  --n-paths-price 131072 --n-paths-greeks 131072 --n-steps 128
```
