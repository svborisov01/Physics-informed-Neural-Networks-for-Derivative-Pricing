# 2-factor Bergomi results

Checkpoint: `checkpoints/bergomi2.pt`  
Training uses `flat_fwd` BS baseline, relative PDE residual, short-τ sampling,
and a soft **MC price anchor** grid.

See [`SUMMARY.md`](SUMMARY.md) for metrics and
[`SURFACE_COMPARISON.md`](SURFACE_COMPARISON.md) for IV-surface prose.

```bash
python scripts/train_bergomi2.py --epochs 1000
python scripts/test_bergomi2.py --checkpoint checkpoints/bergomi2.pt
```
