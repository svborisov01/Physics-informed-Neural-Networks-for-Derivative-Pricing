# 2-factor Bergomi results

Checkpoint: `checkpoints/bergomi2.pt`  
Feature set **`v1b`** (34 features), hidden=96, depth=4.
Training uses `flat_fwd` BS baseline, relative PDE residual, short-τ sampling,
and a soft **MC price anchor** grid.

See [`SUMMARY.md`](SUMMARY.md) for metrics / hillclimb and
[`SURFACE_COMPARISON.md`](SURFACE_COMPARISON.md) for IV-surface prose.

```bash
python scripts/hillclimb_bergomi2.py --epochs 500 --full-epochs 1400
python scripts/train_bergomi2.py --feature-set v1b --hidden 96 --depth 4 --epochs 1400
python scripts/test_bergomi2.py --checkpoint checkpoints/bergomi2.pt
```
