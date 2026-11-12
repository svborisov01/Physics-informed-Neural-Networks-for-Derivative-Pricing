# 2-factor Bergomi results

Checkpoint: `checkpoints/bergomi2.pt`  
Prose surface comparison: [`SURFACE_COMPARISON.md`](SURFACE_COMPARISON.md)  
Numeric summary: [`SUMMARY.md`](SUMMARY.md), [`test_report.json`](test_report.json)

## Reproduce

```bash
python scripts/train_bergomi2.py --epochs 1200
python scripts/test_bergomi2.py --checkpoint checkpoints/bergomi2.pt
python scripts/plot_vol_surface_richness.py --out-dir results/bergomi2
```
