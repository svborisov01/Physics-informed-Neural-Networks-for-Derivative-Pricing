# 2-Factor Bergomi PINN

Implementation of the **classical two-factor Bergomi** model as a
physics-informed neural network, following the same BS-correction + softplus
pattern as the 1-factor Bergomi / Heston modules.

## Dynamics

\[
\begin{aligned}
\frac{dS}{S} &= r\,dt + \sqrt{v}\,dW^{S}, \\
dX_1 &= -\kappa_1 X_1\,dt + dW^{1}, \\
dX_2 &= -\kappa_2 X_2\,dt + dW^{2},
\end{aligned}
\]

with correlations \((\rho_1,\rho_2,\rho_{12})\) among \((W^{S},W^{1},W^{2})\).

Stationary variance (flat initial forward variance \(\xi_0\)):

\[
v(X_1,X_2)
=
\xi_0
\exp\!\Big(
\omega_1 X_1 + \omega_2 X_2
- \tfrac{\omega_1^{2}}{4\kappa_1}
- \tfrac{\omega_2^{2}}{4\kappa_2}
- \tfrac{\omega_1\omega_2\rho_{12}}{\kappa_1+\kappa_2}
\Big).
\]

Typically \(\kappa_1 \gg \kappa_2\): **fast** factor shapes short-dated skew/smile;
**slow** factor keeps longer-dated skew alive.

## Optimal feature set (28)

Chosen to mirror the validated 1-factor map while adding the second OU factor
and both mean-reversion clocks:

| Group | Features |
|-------|----------|
| State | \(x,X_1,X_2,\tau\) (normalized) |
| Params | \(r,\xi_0,\omega_1,\omega_2,\kappa_1,\kappa_2,\rho_1,\rho_2,\rho_{12}\) |
| Time basis | \(\tau^{2},\sin(\pi\tau),\cos(\pi\tau)\) |
| Variance drive | stationary \(\log v\) drive |
| Timescales | \(\mathrm{mr_{fast}}=1-e^{-\kappa_1\tau}\), \(\mathrm{mr_{slow}}=1-e^{-\kappa_2\tau}\) |
| Nonlinear / cross | \(x^{2},\|x\|,X_1^{2},X_2^{2},x\tau,xX_1,xX_2,X_1X_2\) |
| Put/call | \(\mathrm{cp}\) |

Architecture default: **width 96 × depth 4**, softplus \(\beta=400\),
\(u=\mathrm{softplus}_\beta(u_{\mathrm{BS}}+U)\).

## Files

| Path | Role |
|------|------|
| `pricing/bergomi2_option_pricing.py` | PDE residual, PINN, training |
| `support_tools/monte_carlo_pricing_tools.py` → `Bergomi2_Monte_Carlo` | Ground truth |
| `scripts/train_bergomi2.py` | Training entrypoint |
| `scripts/test_bergomi2.py` | PINN vs MC grid test |
| `scripts/plot_vol_surface_richness.py` | BS / Heston / 1F / 2F IV comparison |
| `trained_models/bergomi2.pt` | Checkpoint |
| `bergomi2/artifacts/` | Losses, error heatmaps, IV plots, reports |

## Reproduce

```bash
python scripts/train_bergomi2.py --epochs 2500
python scripts/test_bergomi2.py --checkpoint trained_models/bergomi2.pt
python scripts/plot_vol_surface_richness.py --out-dir bergomi2/artifacts
```

## Volatility-surface richness (intuition)

See `artifacts/vol_smile_richness.png` and friends:

1. **Black–Scholes** — constant \(\sigma\): flat smile, flat term structure.
2. **Heston** — one CIR variance factor: skew + smile, but a **single**
   mean-reversion speed; skew decays at one rate.
3. **1-factor Bergomi** — one OU factor on log-forward-variance: similar
   dimensionality to Heston, often stronger short wings (lognormal variance).
4. **2-factor Bergomi** — fast + slow OU factors: independently tunable
   short-dated and long-dated skew / ATM term structure → richer IV surface.

## Results

See `artifacts/test_report.json` and `artifacts/SUMMARY.md` after training/tests.
