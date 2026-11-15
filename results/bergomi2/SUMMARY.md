# 2-Factor Bergomi PINN — results

## Training (current checkpoint)

| Item | Value |
|------|-------|
| Checkpoint | `checkpoints/bergomi2.pt` |
| Feature set | **`v1b`** (28 core + 6 curated: √τ, κ-ratio, skew proxies, x·mr) |
| Architecture | hidden=**96**, depth=**4**, **34 features** |
| Positivity | softplus β=400 |
| Baseline | BS with **`flat_fwd`** σ_eff |
| PDE loss | relative to \|L[u_BS]\| + Huber; λ_pde anneal |
| Extras | **MC price anchors** (36-point grid), short-τ collocation bias |
| Epochs | 1400 (best @ 1114) |
| Best selection loss | ≈ **4.13×10⁻⁴** |

Previous production weights archived as
`checkpoints/legacy/bergomi2_h96_d4_f28.pt` (28-feature `v1`).

## Hillclimb (feature sets × width/depth)

Sweep script: `scripts/hillclimb_bergomi2.py`.
Short runs (500 epochs) ranked by RMSE on a fixed MC grid
(\(S\in\{80..130\}\), \(\tau\in[0.25,1.5]\times6\), 40k paths, seed 0):

| Rank | Config | RMSE | Max \|err\| |
|------|--------|------|-------------|
| legacy | v1 h96 d4 (archived) | 0.1977 | 0.3318 |
| **1** | **v1b h96 d4** | **0.2230** | **0.3942** |
| 2 | v1b h128 d3 | 0.2263 | 0.4356 |
| 3 | v1 h160 d4 | 0.2388 | 0.4675 |
| … | v2 (40 feats) | ≥0.31 | — |

Full train of the winner (**v1b / 96 / 4**, 1400 epochs) on the same grid:

| Metric | Legacy v1 | **This run (v1b)** |
|--------|-----------|--------------------|
| RMSE | 0.1977 | **0.1748** |
| MAE | 0.1762 | **0.1535** |
| Max \|err\| | 0.3318 | **0.2926** |
| ATM \(T=1\) PINN | 10.286 | 10.292 |
| ATM \(T=1\) MC | 10.122 | 10.122 |

Takeaway: the **full** v2 feature dump hurt accuracy; a **small curated**
extra set (`v1b`) plus the same width/depth as before wins after longer training.
Wider/deeper nets were not reliably better at 500 epochs.

## Accuracy vs Monte Carlo (official scripts)

Market: \(\xi_0=0.04\), \((\omega_1,\omega_2)=(1.0,0.6)\),
\((\kappa_1,\kappa_2)=(4.0,0.8)\), \((\rho_1,\rho_2,\rho_{12})=(-0.6,-0.3,0.2)\),
\(r=0.05\), \(X_1=X_2=0\), \(K=100\).

### Coarse slice (`scripts/test_bergomi2.py`)

Grid: \(S\in\{80,\ldots,130\}\), \(\tau\in[0.25,1.5]\) (6 mats), MC 40k paths.

| Metric | Prior (v1) | This run (v1b) |
|--------|------------|----------------|
| RMSE | ≈ 0.31 | ≈ **0.278** |
| MAE | ≈ 0.25 | ≈ **0.229** |
| Max \|err\| | ≈ 0.59 | ≈ **0.516** |
| ATM \(T=1\) PINN | 10.29 | **10.29** |
| ATM \(T=1\) MC | 10.11 ± 0.04 | 10.11 ± 0.04 |

### Fine grid (matched to 1-factor smile heatmaps)

\(S\in[70,140]\) step 2.5, \(\tau\in[0.25,1.5]\) (21 pts), shared-path MC 40k.

| Metric | Prior (v1) | This run (v1b) | in \(u=V/K\) |
|--------|------------|----------------|--------------|
| RMSE | 0.2030 | **0.1801** | **0.0018** |
| MAE | 0.1712 | **0.1508** | 0.0015 |
| Max \|err\| | 0.4640 | **0.4235** | 0.0042 |

Plot: `abs_err_heatmap.png`.
Regenerate with `python scripts/plot_bergomi2_heatmap_1f_grid.py`.

## Reproduce

```bash
# architecture + feature sweep, then full-train winner
python scripts/hillclimb_bergomi2.py --epochs 500 --full-epochs 1400

# or train a chosen config directly
python scripts/train_bergomi2.py --feature-set v1b --hidden 96 --depth 4 --epochs 1400
python scripts/test_bergomi2.py --checkpoint checkpoints/bergomi2.pt
```

## Artifacts

Hillclimb dumps: `results/bergomi2/hillclimb/`.
Stylized IV smiles and [`SURFACE_COMPARISON.md`](SURFACE_COMPARISON.md).
