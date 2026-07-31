# 1-Factor Bergomi PDE (log-moneyness form)

Design notes for PINN pricing under the classical **one-factor Bergomi** model
(Markovian forward-variance factor; not rough Bergomi).

---

## 1. Model dynamics

Under the risk-neutral measure, with dividend yield \(q\):

\[
\begin{aligned}
\frac{dS_t}{S_t}
&= (r - q)\,dt + \sqrt{v_t}\,dW_t^{S}, \\[4pt]
dX_t
&= -\kappa X_t\,dt + dW_t^{X}, \\[4pt]
d\langle W^{S}, W^{X}\rangle_t
&= \rho\,dt.
\end{aligned}
\]

Here \(X_t\) is an Ornstein–Uhlenbeck forward-variance factor. Instantaneous variance
is the diagonal of the forward variance curve:

\[
v_t = \xi_t^{t}.
\]

### Flat initial forward variance

Assume \(\xi_0^{T} \equiv \xi_0 > 0\) (constant). The exponential-martingale
construction of Bergomi then gives

\[
\xi_t^{T}
=
\xi_0
\exp\!\Big(
\omega\, e^{-\kappa(T-t)} X_t
-
\tfrac{\omega^{2}}{2}
\int_{0}^{t} e^{-2\kappa(T-s)}\,ds
\Big),
\]

and therefore the spot variance

\[
\boxed{
v(t,X)
=
\xi_0
\exp\!\Big(
\omega X
-
\tfrac{\omega^{2}}{4\kappa}\bigl(1 - e^{-2\kappa t}\bigr)
\Big).
}
\]

**Parameters**

| Symbol | Meaning |
|--------|---------|
| \(\xi_0\) | Initial flat forward variance |
| \(\omega\) | Vol-of-variance (Bergomi) |
| \(\kappa\) | Mean-reversion speed of \(X\) |
| \(\rho\) | Spot–factor correlation |
| \(r, q\) | Rates |

**Notes**

- \(X\) is Gaussian OU; typically \(X_0 = 0\).
- As \(\omega \to 0\), \(v \to \xi_0\) and the model collapses to Black–Scholes with \(\sigma = \sqrt{\xi_0}\).
- \(v(t,X)\) depends on **calendar time** \(t\) through the variance of \(X\), so the pricing PDE is time-inhomogeneous even for a flat initial curve.

### Stationary approximation (optional)

Replacing \(\mathrm{Var}(X_t)\) by its stationary value \(1/(2\kappa)\) removes
explicit \(t\)-dependence:

\[
v_{\mathrm{stat}}(X)
=
\xi_0
\exp\!\Big(
\omega X - \tfrac{\omega^{2}}{4\kappa}
\Big).
\]

This is convenient for PINNs (same dimensional structure as Heston in \((x,v)\) /
\((x,X)\)) and is a reasonable modelling choice when training over a maturity
window. The exact PDE below keeps calendar time; the PINN implementation can
switch between the two via a flag.

---

## 2. Pricing PDE in \((t,S,X)\)

Let \(V(t,S,X)\) be a European option price. Itô’s formula yields

\[
\begin{aligned}
\partial_t V
&+ (r-q)S\,\partial_S V
+ \tfrac12 v(t,X)\,S^{2}\,\partial_{SS} V \\
&- \kappa X\,\partial_X V
+ \tfrac12\,\partial_{XX} V
+ \rho\sqrt{v(t,X)}\,S\,\partial_{SX} V
- r V
= 0,
\end{aligned}
\]

with terminal condition \(V(T,S,X) = g(S)\) (e.g. \(\max(S-K,0)\)).

---

## 3. Change of variables: \(x = \log(S/K)\), \(\tau = T-t\), \(u = V/K\)

Define

\[
x = \log(S/K),
\qquad
\tau = T - t,
\qquad
u(\tau, x, X) = \frac{V(T-\tau,\, K e^{x},\, X)}{K}.
\]

Standard chain-rule identities:

\[
\begin{aligned}
S\,\partial_S V &= K\,\partial_x u, \\
S^{2}\,\partial_{SS} V &= K\bigl(\partial_{xx} u - \partial_x u\bigr), \\
S\,\partial_{SX} V &= K\,\partial_{xX} u, \\
\partial_t V &= -K\,\partial_\tau u.
\end{aligned}
\]

Substituting and dividing by \(K\) gives the **normalized 1-factor Bergomi PDE**:

\[
\boxed{
\begin{aligned}
\partial_\tau u
&=
\bigl(r - q - \tfrac12 v\bigr)\,\partial_x u
+ \tfrac12 v\,\partial_{xx} u \\
&\quad
- \kappa X\,\partial_X u
+ \tfrac12\,\partial_{XX} u
+ \rho\sqrt{v}\,\partial_{xX} u
- r\, u,
\end{aligned}
}
\]

where

\[
v = v(T-\tau,\, X)
=
\xi_0
\exp\!\Big(
\omega X
-
\tfrac{\omega^{2}}{4\kappa}\bigl(1 - e^{-2\kappa(T-\tau)}\bigr)
\Big)
\]

(exact calendar-time form), or \(v = v_{\mathrm{stat}}(X)\) under the stationary
approximation.

Equivalently, residual form used by the PINN:

\[
\begin{aligned}
\mathcal{R}[u]
&=
-\partial_\tau u
+ \bigl(r - q - \tfrac12 v\bigr)\,\partial_x u
+ \tfrac12 v\,\partial_{xx} u \\
&\quad
- \kappa X\,\partial_X u
+ \tfrac12\,\partial_{XX} u
+ \rho\sqrt{v}\,\partial_{xX} u
- r\, u
\;=\; 0.
\end{aligned}
\]

### Terminal condition

\[
u(0, x, X)
=
\begin{cases}
\mathrm{ReLU}(e^{x} - 1), & \text{call}, \\
\mathrm{ReLU}(1 - e^{x}), & \text{put}.
\end{cases}
\]

### Spatial asymptotics (calls; \(q = 0\))

Same leading behaviour as Black–Scholes / Heston in \(x\)-space:

\[
\begin{aligned}
x &\to -\infty: & u &\to 0, & \partial_x u &\to 0, \\
x &\to +\infty: & u &\sim e^{x} - e^{-r\tau}, & \partial_x u &\sim e^{x}.
\end{aligned}
\]

Puts swap the two ends. Boundaries in \(X\) can use Neumann (vanishing \(X\)-derivative)
or soft softplus / decay regularisation, since \(X\) is Gaussian and large \(|X|\) is rare.

---

## 4. BS correction: is it practically feasible?

**Yes — and it is the recommended architecture**, for the same reasons as Heston.

### Parallel with Heston

| | Heston | 1-factor Bergomi |
|--|--------|------------------|
| Markov state | \((S, v)\) | \((S, X)\) (or \((S,v)\) via \(v(t,X)\)) |
| Vol-of-vol | \(\sigma\sqrt{v}\,dW^{v}\) | \(\omega\) in \(\log v\) |
| BS limit | \(\sigma\to 0\) ⇒ BS(\(\sqrt{v}\)) locally | \(\omega\to 0\) ⇒ **exact** BS(\(\sqrt{\xi_0}\)) |
| Dimensionality | 2 spatial + \(\tau\) | 2 spatial + \(\tau\) |

Write

\[
u = u_{\mathrm{BS}} + U,
\]

and train a network only for the correction \(U\), enforcing the full Bergomi
operator on \(u_{\mathrm{BS}} + U\) (identical residual strategy to
`heston_option_pricing.py`).

### Choice of BS baseline \(\sigma_{\mathrm{eff}}\)

Three practical options (mirroring `sigma_bs_effective` in Heston):

1. **`flat_fwd`** (recommended default)
   \[
   \sigma_{\mathrm{eff}} = \sqrt{\xi_0}.
   \]
   Independent of \(X\). Exact when \(\omega = 0\). Correction \(U\) absorbs all
   stochastic-vol smile/skew. Terminal condition: \(u_{\mathrm{BS}}\to\) payoff and
   \(U\to 0\) at \(\tau = 0\).

2. **`spot_var`**
   \[
   \sigma_{\mathrm{eff}} = \sqrt{v(t,X)}.
   \]
   Tracks local variance; useful near expiry. \(u_{\mathrm{BS}}\) still depends on
   \(X\) through \(v\), so autograd through the baseline is required (as in Heston).

3. **`hybrid`**
   Blend of flat forward and spot variance with a \(\tau\)-dependent weight
   (same idea as Heston hybrid mode).

Because \(\omega\to 0\) recovers BS **globally** (not only locally), option 1 is
cleaner than the Heston case and should be the first implementation target.

### Why this is feasible for PINNs

- Same residual structure as the existing Heston PINN: learn \(U\), apply the PDE
  operator to \(u_{\mathrm{BS}}+U\), scale residual by \(1+|u_{\mathrm{BS}}|\).
- Feature map can reuse Heston-style inputs with \(X\) (or \(\log(v/\xi_0)\)) replacing
  \((v,\kappa,\theta,\sigma)\), plus Bergomi parameters \((\xi_0,\omega,\kappa,\rho)\).
- Domain for \(X\): e.g. \(X\in[-X_{\max}, X_{\max}]\) with \(X_{\max}\approx 3/\sqrt{2\kappa}\)
  (covers most OU mass). Softplus / tanh scaling on \(U\) as in Heston.
- Ground truth for testing: no simple closed form; use **Monte Carlo** (QE-style
  or exact OU + Euler for \(S\)) or Fourier methods when available. COS is not as
  standard as for Heston; MC is the natural first benchmark.

### Caveats

1. **Calendar time in \(v(t,X)\)**  
   Training over many maturities with the exact formula needs \(T\) (or \(t\)) as an
   input, or use \(v_{\mathrm{stat}}\). Prefer starting with the stationary variance
   for a direct Heston-like PINN, then add calendar-time dependence if needed.

2. **Parameter ranges**  
   Large \(\omega\) or \(|\rho|\) near 1 makes \(v\) extremely skewed; keep \(\omega\)
   moderate in the first training runs (e.g. \(\omega\in(0,2]\) for \(\xi_0\sim 0.04\)).

3. **No American constraint planned** for Bergomi v1 (European only), matching
   current Heston training.

---

## 5. Recommended PINN residual (stationary Bergomi + BS correction)

With \(q = 0\), \(v = v_{\mathrm{stat}}(X)\), and \(u = u_{\mathrm{BS}}(x,\tau;r,\sigma_{\mathrm{eff}}) + U\):

\[
\begin{aligned}
\mathcal{R}
&=
-\partial_\tau u
+ \bigl(r - \tfrac12 v\bigr)\,\partial_x u
+ \tfrac12 v\,\partial_{xx} u \\
&\quad
- \kappa X\,\partial_X u
+ \tfrac12\,\partial_{XX} u
+ \rho\sqrt{v}\,\partial_{xX} u
- r\, u.
\end{aligned}
\]

Default baseline: \(\sigma_{\mathrm{eff}} = \sqrt{\xi_0}\) (`flat_fwd`).

---

## 6. Suggested implementation plan (next steps)

1. Add `pricing/bergomi_option_pricing.py` mirroring the Heston module:
   - `v_bergomi(X, xi0, omega, kappa, t=None, stationary=True)`
   - `pde_dynamic_x`, terminal / boundary losses on \(u = u_{\mathrm{BS}} + U\)
   - `PINN` learning \(U(x,X,\tau,\ldots)\)
2. Extend `model_wrapper.py` with `ModelType.BERGOMI`.
3. Benchmark against Monte Carlo paths of \((S,X)\).
4. Only then consider multi-factor / rough Bergomi (non-Markovian — much harder).

---

## References

- Bergomi, L. *Stochastic Volatility Modeling* (Wiley), Ch. on forward variance.
- Bergomi, L. & Guyon, J. (2012). Stochastic volatility’s orderly smiles.
- Fang & Oosterlee COS method remains available for Heston benchmarks; for Bergomi
  use MC as primary ground truth.
