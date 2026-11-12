# Physics-Informed Neural Networks for Derivative Pricing

PINNs for European option pricing in log-moneyness space \(x=\log(S/K)\),
with normalized prices \(u=V/K\). Models range from constant-vol Black–Scholes
to one- and two-factor Bergomi forward-variance dynamics.

## Test results

Numbers below are from the shipped checkpoints under `checkpoints/`,
reproduced via `support_tools.model_wrapper.run_slice_test` (or the Bergomi
QMC/MC suites). Full dumps live in `results/metrics.json`.

| Model | Checkpoint | Benchmark | RMSE | MAE | Max \|err\| | Notes |
|-------|------------|-----------|------|-----|-------------|-------|
| 2D Black–Scholes | `two_d.pt` | BS closed form | **0.238** | 0.192 | 1.60 | ATM \(T=1\): 10.68 |
| HD Black–Scholes | `hd_minimal.pt` | BS closed form | **0.301** | 0.237 | 1.22 | ATM \(T=1\): 10.19 |
| Heston | `long_training_heston.pt` | COS method | **0.216** | 0.172 | 0.44 | Default test market |
| 1-factor Bergomi | `bergomi.pt` | QMC+ADD (131k) | **0.137** | 0.110 | 0.32 | Rel. RMSE 0.061; 0 negatives |
| 2-factor Bergomi | `bergomi2.pt` | MC (40k paths) | **0.309** | 0.255 | 0.59 | ATM $T=1$: 10.29 vs MC 10.11 |

1-factor Bergomi smile heatmaps (abs \(L^\infty\) vs QMC+ADD): baseline 0.30,
strong vol-of-vol 0.51, strong neg. skew 0.59, mild fast MR 0.21, slow MR 0.92,
near-BS 0.17 — see `results/bergomi/SUMMARY.md`.

Plots and reports: `results/<model>/`.

## Stochastic volatility demo

Fit BS / Heston / 1F / **2F Bergomi** IV surfaces with honest single- vs
two-clock calibration — see [`stochastic_volatility_demo/`](stochastic_volatility_demo/):

```bash
python stochastic_volatility_demo/run_demo.py            # synthetic two-timescale (default)
python stochastic_volatility_demo/run_demo.py ^SPX       # live strip
```

## Layout

```
├── pricing/              # PINN models + training loops
├── support_tools/        # load / test / MC / analytics
├── scripts/              # CLI train / test / plot entrypoints
├── checkpoints/          # active .pt weights (legacy/ for old ones)
├── results/              # per-model test metrics & plots
├── notebooks/            # demos (+ archive/)
├── stochastic_volatility_demo/  # market IV fit (2F advantage)
├── README.md
└── requirements.txt
```

| Path | Role |
|------|------|
| `pricing/two_d_option_pricing.py` | 2D BS, fixed \(r,\sigma\) |
| `pricing/hd_option_pricing.py` | HD BS, variable \(r,\sigma\) |
| `pricing/heston_option_pricing.py` | Heston BS-correction |
| `pricing/bergomi_option_pricing.py` | 1-factor Bergomi |
| `pricing/bergomi2_option_pricing.py` | 2-factor Bergomi |
| `support_tools/model_wrapper.py` | Universal load / price / test / Greeks |
| `results/bergomi2/SURFACE_COMPARISON.md` | IV-surface richness (prose) |

## Install

```bash
git clone <repo-url>
cd Physics-informed-Neural-Networks-for-Derivative-Pricing
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.0+. Run commands from the repo root.

## Quick start

```python
from support_tools.model_wrapper import load_model, run_slice_test, predict_price

pinn, meta = load_model("checkpoints/two_d.pt")
print(meta["model_type"])  # two_d

results = run_slice_test(pinn, return_values=True, K=100, r=0.05, sigma=0.2)
print(predict_price(pinn, S=100, K=100, tau=1.0))
```

### Reproduce model tests

```bash
# 2D / HD / Heston via the wrapper (example)
python - <<'PY'
from support_tools.model_wrapper import load_model, run_slice_test
for ckpt in ["checkpoints/two_d.pt", "checkpoints/hd_minimal.pt",
             "checkpoints/long_training_heston.pt"]:
    pinn, meta = load_model(ckpt)
    run_slice_test(pinn, print_metrics=True)
PY

# 1-factor Bergomi extensive suite (slow)
python scripts/run_bergomi_extensive_tests.py --checkpoint checkpoints/bergomi.pt

# 2-factor Bergomi MC grid
python scripts/test_bergomi2.py --checkpoint checkpoints/bergomi2.pt
```

### Train

```python
import torch
from pricing.two_d_option_pricing import PINN, train_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pinn = PINN(x_min=-5, x_max=5, T=4.0, r=0.05, sigma=0.2, hidden=50, depth=5).to(device)
history = train_network(pinn, epochs=5000, best_model_path="checkpoints/my_model.pt")
```

```bash
python scripts/train_bergomi2.py --epochs 1200
```

Walkthroughs: `notebooks/demo_all_models.ipynb`, `notebooks/demo_full.ipynb`.

## Checkpoints

| File | Status |
|------|--------|
| `checkpoints/two_d.pt` | Active |
| `checkpoints/hd_minimal.pt` | Active |
| `checkpoints/long_training_heston.pt` | Active |
| `checkpoints/bergomi.pt` | Active (softplus β=400) |
| `checkpoints/bergomi2.pt` | Active (softplus β=400) |
| `checkpoints/legacy/*` | Old feature sets / backups — do not use with current code |

## Notes

- Heston / Bergomi learn a correction \(U\) on a Black–Scholes baseline;
  positivity uses softplus \(\beta=400\) for Bergomi models.
- Heston ground truth: Fang–Oosterlee COS. Bergomi ground truth: MC / QMC+ADD.
- American constraints exist in BS modules but are off in the demos
  (`lambda_american=0`).

## License

Academic / thesis work. See repository for authorship details.
