# Heston PDE in log-moneyness

Normalized European Heston pricing PDE after the same change of variables used
for Black–Scholes in this project: \(x=\log(S/K)\), \(\tau=T-t\), \(u=V/K\).

---

## 1. Black–Scholes reference (log-moneyness)

With \(x=\log(S/K)\), \(\tau=T-t\), \(u=V/K\), and constant volatility \(\sigma\):

\[
u_{\tau}
-
\tfrac12\sigma^{2}\,u_{xx}
-
\bigl(r-\tfrac12\sigma^{2}\bigr)\,u_{x}
+ r\,u
= 0.
\]

---

## 2. Heston dynamics

Under the risk-neutral measure (dividend yield \(q\)):

\[
\begin{aligned}
\frac{dS_t}{S_t}
&= (r-q)\,dt + \sqrt{v_t}\,dW_t^{S}, \\[4pt]
dv_t
&= \kappa(\bar\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^{v}, \\[4pt]
d\langle W^{S}, W^{v}\rangle_t
&= \rho\,dt.
\end{aligned}
\]

| Symbol | Meaning |
|--------|---------|
| \(v\) | Instantaneous variance |
| \(\kappa\) | Mean-reversion speed |
| \(\bar\theta\) | Long-run variance (written \(\bar\theta\) to avoid clashing with other \(\theta\) labels) |
| \(\sigma\) | Vol-of-vol |
| \(\rho\) | Spot–variance correlation |

---

## 3. Pricing PDE in \((t,S,v)\)

For a European price \(V(t,S,v)\):

\[
\begin{aligned}
\partial_t V
&+ (r-q)S\,\partial_S V
+ \kappa(\bar\theta-v)\,\partial_v V \\
&+ \tfrac12 v\,S^{2}\,\partial_{SS} V
+ \rho\sigma v\,S\,\partial_{Sv} V
+ \tfrac12\sigma^{2} v\,\partial_{vv} V
- r V
= 0,
\end{aligned}
\]

with \(V(T,S,v)=g(S)\).

---

## 4. Change of variables

\[
x = \log(S/K),
\qquad
\tau = T-t,
\qquad
u(\tau,x,v) = \frac{V(T-\tau,\,Ke^{x},\,v)}{K}.
\]

Chain-rule identities:

\[
\begin{aligned}
\partial_t V &= -K\,\partial_\tau u, \\
S\,\partial_S V &= K\,\partial_x u, \\
S^{2}\,\partial_{SS} V &= K\bigl(\partial_{xx} u - \partial_x u\bigr), \\
S\,\partial_{Sv} V &= K\,\partial_{xv} u, \\
\partial_v V &= K\,\partial_v u, \\
\partial_{vv} V &= K\,\partial_{vv} u.
\end{aligned}
\]

Substitute into the \((t,S,v)\) PDE and divide by \(K\):

\[
\begin{aligned}
-\partial_\tau u
&+ (r-q)\,\partial_x u
+ \kappa(\bar\theta-v)\,\partial_v u \\
&+ \tfrac12 v\,(\partial_{xx} u - \partial_x u)
+ \rho\sigma v\,\partial_{xv} u
+ \tfrac12\sigma^{2} v\,\partial_{vv} u
- r\,u
= 0.
\end{aligned}
\]

---

## 5. Heston PDE in log-moneyness (BS-analog form)

Collecting terms — same residual arrangement as the Black–Scholes reference:

\[
\boxed{
\begin{aligned}
u_{\tau}
&-
\tfrac12 v\,u_{xx}
-
\bigl(r-q-\tfrac12 v\bigr)\,u_{x}
-
\rho\sigma v\,u_{xv} \\
&-
\tfrac12\sigma^{2} v\,u_{vv}
-
\kappa(\bar\theta-v)\,u_{v}
+ r\,u
= 0.
\end{aligned}
}
\]

With \(q=0\) (matching the usual training setup in `heston_option_pricing.py`):

\[
\boxed{
\begin{aligned}
u_{\tau}
&-
\tfrac12 v\,u_{xx}
-
\bigl(r-\tfrac12 v\bigr)\,u_{x}
-
\rho\sigma v\,u_{xv} \\
&-
\tfrac12\sigma^{2} v\,u_{vv}
-
\kappa(\bar\theta-v)\,u_{v}
+ r\,u
= 0.
\end{aligned}
}
\]

Compared with Black–Scholes, \(\sigma^{2}\mapsto v\) in the spot terms, and three
variance-channel terms appear: mixed \(u_{xv}\), pure \(u_{vv}\), and the
drift \(u_{v}\).

### Residual form used by the PINN

The implementation in `pde_dynamic_x` enforces the opposite-sign residual
\(\mathcal{R}[u]=0\):

\[
\begin{aligned}
\mathcal{R}[u]
&=
-\partial_\tau u
+ \bigl(r-\tfrac12 v\bigr)\,\partial_x u
+ \tfrac12 v\,\partial_{xx} u \\
&\quad
+ \rho\sigma v\,\partial_{xv} u
+ \tfrac12\sigma^{2} v\,\partial_{vv} u
+ \kappa(\bar\theta-v)\,\partial_v u
- r\,u.
\end{aligned}
\]

(In code, long-run variance is the parameter `theta`.)

### Terminal condition

\[
u(0,x,v)
=
\begin{cases}
\mathrm{ReLU}(e^{x}-1), & \text{call}, \\
\mathrm{ReLU}(1-e^{x}), & \text{put}.
\end{cases}
\]

---

## References

- Heston, S. (1993). A closed-form solution for options with stochastic volatility.
- Implementation residual: `pricing/heston_option_pricing.py` → `heston_operator`.
