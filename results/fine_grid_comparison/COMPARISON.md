# Fine-grid comparison (same (S, tau) lattice)

Grid: S in [70, 140] step 2.5, tau in [0.25, 1.50] x 21, K=100.

| Model | Benchmark | Training aid | RMSE (u) | Max (u) | RMSE (V) | Max (V) |
|-------|-----------|--------------|----------|---------|----------|---------|
| Heston | COS | — | 0.0022 | 0.0039 | 0.2202 | 0.3910 |
| Bergomi 1F | QMC+ADD | — | 0.0015 | 0.0030 | 0.1517 | 0.3019 |
| Bergomi 2F | QMC+ADD | — | 0.0026 | 0.0047 | 0.2578 | 0.4663 |
| Bergomi 2F + QMC aid | QMC+ADD | QMC+ADD samples | 0.0018 | 0.0034 | 0.1790 | 0.3446 |

