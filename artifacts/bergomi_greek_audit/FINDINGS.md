# Bergomi Greek smoothness audit

Checkpoint: `trained_models/bergomi.pt`  
Smile: baseline \(\xi_0=0.04,\omega=1,\kappa=2,\rho=-0.5,X=0\)  
Artifacts: `artifacts/bergomi_greek_audit/`

## 1) Are the Greeks “really” that smooth?

**Short answer: yes they are smooth — mostly because they are almost Black–Scholes.**

### Internal consistency
- Autograd **Delta / Theta** match central finite differences of the PINN price to \(\sim 10^{-4}\)–\(10^{-3}\).
- Autograd **Gamma** is trustworthy; naive FD Gamma from float32 prices is unstable for tiny \(h\) (classic second-difference cancellation). With moderate \(h\) away from ATM it matches; ATM needs autograd.

### Compared to the BS baseline inside the model
With `spot_var` baseline \(\sigma=\sqrt{v(X)}\approx 0.188\):

| Greek | L1(\|PINN−BS\|) / L1(\|BS\|) | discrete-Laplacian RMS PINN vs BS |
|-------|------------------------------|-----------------------------------|
| Delta | **1.2%** | same order |
| Gamma | **1.4%** | same order |
| Theta | ~29% | PINN slightly *smoother* than BS |

Near-expiry ATM Gamma peak ratio PINN/BS ≈ **0.99** for \(\tau\in[0.05,1.4]\).

So the impressive S-shaped Delta / peaked Gamma / Theta valley are essentially the **closed-form BS baseline**. The network correction \(U\) only nudges them.

### Compared to Bergomi QMC+ADD (this is the suspicious part)
At ATM \(S=100\):

| τ | Δ PINN | Δ BS | Δ QMC+ADD | \|PINN−MC\| | \|MC−BS\| |
|---|--------|------|-----------|------------|-----------|
| 0.4 | 0.592 | 0.590 | 0.614 | 0.022 | 0.024 |
| 1.0 | 0.646 | 0.641 | 0.666 | 0.019 | 0.025 |
| 1.4 | 0.673 | 0.665 | 0.688 | 0.016 | 0.023 |

PINN Delta stays glued to BS; Bergomi MC wants a **~0.02–0.025** larger ATM Delta. Smoothness is real, but it is **under-corrected for Bergomi skew / vol-of-vol**.

### Price decomposition \(V=V_{BS}+K\cdot U\)
Typical \(|K\cdot U|\sim 0.2\)–\(0.4\) (max ~0.39 on the test grid).  
Capacity under current head (`0.15 * … * tanh * ω/ω_max`) with \(\omega=1,\omega_{max}=2\) allows up to ~\$7.5 ATM — so capacity is not saturated; the PDE residual optimum just keeps \(U\) small.

At some OTM points PINN is **worse** than bare BS vs MC (e.g. \(S=85,\tau=1.4\): \|BS−MC\|≈0.01, \|PINN−MC\|≈0.15). ATM PINN does improve over BS.

### Why this happens (architecture)
\[
u = u_{BS}(x,\tau;r,\sigma_{\mathrm{eff}}(v)) + U,\quad
U = \frac{\omega}{\omega_{\max}}\cdot(\text{small scale})\cdot\tanh(\mathrm{NN})
\]
- Greeks inherit analytic smoothness from \(u_{BS}\).
- \(\omega\)-gate + small `corr_scale` + `tanh` bias the model toward “BS + tiny bump”.
- PDE collocation without price labels does not strongly force Bergomi skew into Delta.

---

## 2) Ways to make error norms smaller

Ordered from highest expected impact / effort for this codebase:

### A. Give the correction more useful capacity (high impact)
1. **Richer baseline than flat/spot BS**  
   Use a hybrid / Bergomi-short-time asymptotic / effective forward variance so \(U\) only learns residual skew/kurtosis.
2. **Relax the head slightly**  
   Increase `corr_scale`, soften \(\omega\)-gate (e.g. \(\sqrt{\omega/\omega_{\max}}\) or learnable gate), or replace hard `tanh` with a soft residual so ATM Delta can move ~0.02 toward MC.
3. **Positivity constraint on total price**  
   Softplus/relu on \(V\) for calls to kill negative deep-OTM prices (currently \(V\approx -0.07\) at \(S=70,\tau=0.3\)).

### B. Better training signal (high impact)
4. **Residual adaptive resampling**  
   Periodically resample collocation where \(|\mathcal{L}u|\) or \(|x|\) (ATM) is large; already half ATM — push further (e.g. 70% near \(x=0\), extra weight on large \(\omega\), small \(\kappa\)).
5. **Curriculum on smile strength**  
   Train \(\omega\) from small→large / \(\kappa\) from large→small so the net first locks BS, then learns skew.
6. **Semi-supervised / multi-fidelity loss**  
   Add a small number of QMC+ADD price anchors (and optionally pathwise Delta) on a coarse \((S,\tau)\) grid; keep PDE as the main regularizer. This directly attacks the ATM Delta gap.
7. **Longer / larger model**  
   Deeper/wider net + more epochs with cosine restarts; current checkpoint is short relative to Heston runs in the repo.

### C. Loss and PDE details (medium impact)
8. **Rebalance weights**  
   Raise `lambda_terminal` / boundary near ATM; add a mild **Gamma/Delta matching** term vs BS only as \(\omega\to 0\) (already partly implied by omega gate).
9. **Match MC convention**  
   Confirm `stationary=True` vs calendar \(v(t,X)\) against the benchmark; mismatch shows up as maturity-dependent bias.
10. **Float64 for evaluation / mixed precision training**  
    Not for smoothness theater — helps FD diagnostics and tiny residual optimization.

### D. Evaluation hygiene (does not reduce true error, but clarifies norms)
11. Report **PINN−BS** and **BS−MC** alongside **PINN−MC** so “impressive Greeks” are not mistaken for Bergomi accuracy.
12. Use relative norms ATM-weighted: \(\|e\|_1^{w}\) with weight \(\propto \Gamma_{BS}\) or payoff vega.

### Practical “next experiment” shortlist
1. Soft positivity on \(V\) + slightly larger correction scale.  
2. Adaptive ATM / high-\(\omega\) residual sampling + 3–5× longer train.  
3. Add sparse QMC price (+ Delta) anchors in the loss.  
4. Stronger baseline (hybrid σ or Bergomi asymptotic).

Expected: (1)+(2) should cut OTM artifacts and shave L1; (3) is the most direct way to close the ~0.02 ATM Delta / remaining ~\$0.1 price L1 gap.
