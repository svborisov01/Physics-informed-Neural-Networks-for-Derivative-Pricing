# Physics-Informed Neural Networks for Derivative Pricing

Master's thesis project implementing **Physics-Informed Neural Networks (PINNs)** for European option pricing under Black–Scholes and Heston stochastic volatility models.

The network learns normalized option prices \(u = V/K\) in log-moneyness space \(x = \log(S/K)\), trained by minimizing PDE residuals, terminal payoffs, and boundary conditions.

## Models

| Module | Model | Inputs | Description |
|--------|-------|--------|-------------|
| `pricing/two_d_option_pricing.py` | 2D Black–Scholes | \((x, \tau)\) | Fixed interest rate \(r\) and volatility \(\sigma\) |
| `pricing/hd_option_pricing.py` | HD Black–Scholes | \((x, \tau, r, \sigma)\) | Variable \(r\) and \(\sigma\) sampled during training |
| `pricing/heston_option_pricing.py` | Heston correction | 22 features | Learns correction \(U\); total price \(u = u_{BS} + U\) |
| `pricing/bergomi_option_pricing.py` | 1-factor Bergomi | 22 features | BS correction with OU factor \(X\); stationary variance by default |

## Installation

```bash
git clone <repo-url>
cd Physics-informed-Neural-Networks-for-Derivative-Pricing
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+.

## Quick Start

Run all examples from the repository root so imports resolve correctly.

### Load a trained model

```python
from support_tools.model_wrapper import load_model, run_slice_test, visualize_slice_test

pinn, meta = load_model("trained_models/two_d.pt")
print(meta["model_type"])  # "two_d"
```

`load_model` auto-detects the model type and architecture from checkpoint metadata. For custom checkpoints, pass `hidden` and `depth` explicitly.

### Test accuracy (all model types)

```python
from support_tools.model_wrapper import load_model, run_slice_test, visualize_slice_test

# 2D Black–Scholes
pinn, _ = load_model("trained_models/two_d.pt")
results = run_slice_test(pinn, return_values=True, K=100, r=0.05, sigma=0.2)
visualize_slice_test(results, pinn=pinn, values="diff")

# HD Black–Scholes
pinn, _ = load_model("trained_models/hd_minimal.pt")
results = run_slice_test(pinn, return_values=True, r=0.05, sigma=0.2, K=100)

# Heston
pinn, meta = load_model("trained_models/long_training_heston.pt")
results = run_slice_test(
    pinn,
    return_values=True,
    sigma_mode=meta["sigma_mode"],
    K=100,
)
visualize_slice_test(results, pinn=pinn)
```

### Compute Greeks

```python
from support_tools.model_wrapper import load_model, compute_greeks, compare_greeks

pinn, _ = load_model("trained_models/two_d.pt")
price, delta, theta = compute_greeks(pinn, S=100, K=100, tau=1.0)

greeks = compare_greeks(pinn, K=1.0, r=pinn.r, sigma=pinn.sigma)
print("Max |Delta error|:", greeks["delta_diff"].abs().max().item())
```

### Train a new model

```python
import torch
from pricing.two_d_option_pricing import PINN, train_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pinn = PINN(x_min=-5, x_max=5, T=4.0, r=0.05, sigma=0.2, hidden=50, depth=5).to(device)
history = train_network(pinn, epochs=5000, best_model_path="trained_models/my_model.pt")
```

See **`demo_all_models.ipynb`** for a walkthrough of all models including Bergomi (load/train → price → MC test → visualize → Greeks).  
See `demo notebook.ipynb` for full training, comparison, and architecture-search workflows.

## Universal Wrapper API

All testing and visualization goes through `support_tools/model_wrapper.py`:

| Function | Purpose |
|----------|---------|
| `load_model(path)` | Load checkpoint → `(pinn, metadata)` |
| `detect_model_type(pinn)` | Return `"two_d"`, `"hd"`, or `"heston"` |
| `predict_price(pinn, S, K, tau, **kwargs)` | Unified price prediction |
| `run_slice_test(pinn, **kwargs)` | Grid test vs analytical benchmark |
| `visualize_slice_test(results, pinn)` | 3D Plotly surface plot |
| `compute_greeks(pinn, S, K, tau, **kwargs)` | Price, Delta, Theta via autograd |
| `compare_greeks(pinn, **kwargs)` | PINN vs analytic BS Greeks (2D/HD only) |
| `get_default_test_params(pinn)` | Sensible defaults per model type |

Legacy functions `bs_2d_slice_test` and `heston_2d_slice_test` in `support_tools/testing_tools.py` delegate to `run_slice_test` for backward compatibility.

## Pre-trained Checkpoints

| File | Type | Architecture | Compatible |
|------|------|--------------|------------|
| `two_d.pt` | 2D BS | hidden=50, depth=5 | Yes |
| `hd_minimal.pt` | HD BS | hidden=100, depth=3 | Yes |
| `hd_warm_restarts.pt` | HD BS | hidden=40, depth=3 | Yes |
| `long_training_heston.pt` | Heston | hidden=150, depth=6 | Yes |
| `hd.pt` | HD BS (legacy) | 12 input features | **No** — old feature set |
| `heston.pt` | Heston (legacy) | 20 input features | **No** — missing `x_v`, `x_gap` |
| `wide_moneyness_heston.pt` | Heston (legacy) | 20 input features | **No** |

Legacy checkpoints cannot be loaded with the current code. `load_model` raises a clear error if you try.

## Project Structure

```
├── pricing/                    # PINN models and training loops
│   ├── two_d_option_pricing.py
│   ├── hd_option_pricing.py
│   ├── heston_option_pricing.py
│   └── bergomi_option_pricing.py
├── support_tools/
│   ├── model_wrapper.py        # Universal load / test / visualize API
│   ├── testing_tools.py        # Legacy test wrappers
│   ├── graphing_tools.py       # Convergence and 3D plots
│   ├── analytical_pricing_tools.py  # BS and Heston COS ground truth
│   ├── grid_based_pricing_tools.py  # FD Heston pricer (reference)
│   └── monte_carlo_pricing_tools.py # Heston QE + Bergomi MC pricers
├── trained_models/             # Saved checkpoints (.pt)
├── graphs/                     # Pre-rendered comparison plots
├── demo_all_models.ipynb       # Clean demo of all three models via the wrapper
├── demo notebook.ipynb         # Full training / architecture-search notebook
└── old_notebooks/              # Earlier exploratory work
```

## Training Details

- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing (2D) or cosine warm restarts (HD, Heston)
- **Loss**: PDE residual + terminal condition + boundary conditions (+ optional American constraint for BS)
- **Normalization**: Prices in \(u = V/K\), inputs scaled by domain bounds

Heston models use a Black–Scholes baseline with effective volatility (`sigma_bs_effective`) and learn only the correction term \(U\).

Ground-truth Heston prices for testing use the **Fang–Oosterlee COS method** (`heston_price` in `analytical_pricing_tools.py`), not Gil-Pelaez quadrature.

1-factor Bergomi uses the same BS-correction pattern with OU factor \(X\)
(`sigma_bs_bergomi`, default `flat_fwd`). Ground truth for testing is
**Monte Carlo** via `Bergomi_Monte_Carlo` in `monte_carlo_pricing_tools.py`.

```python
from support_tools.monte_carlo_pricing_tools import Bergomi_Monte_Carlo

price, se = Bergomi_Monte_Carlo(
    S0=100, K=100, T=1.0, r=0.05, q=0.0,
    xi0=0.04, omega=1.0, kappa=2.0, rho=-0.5,
    n_paths=100_000, n_steps=100, stationary=True,
    seed=42, return_stderr=True,
)
```

```python
from pricing.bergomi_option_pricing import PINN, train_network

pinn = PINN(
    x_min=-3, x_max=3, X_max=3, T=2.0,
    r_max=0.2, xi0_max=0.1, omega_max=2.0, kappa_max=5.0,
    hidden=128, depth=4,
).to(device)
history = train_network(pinn, epochs=5000, sigma_mode="flat_fwd", stationary=True)
```

## Known Limitations

- American option constraints are implemented but not used in the demo (`lambda_american=0`)
- Greek comparison (`compare_greeks`) is defined for Black–Scholes models only
- Three legacy checkpoints use outdated input feature sets and cannot be loaded
- Runs assume execution from the repository root (no pip-installable package yet)
- 1-factor Bergomi PINN is designed but not yet implemented — see [`docs/bergomi_1factor_pde.md`](docs/bergomi_1factor_pde.md)

## License

Academic / thesis work. See repository for authorship details.
