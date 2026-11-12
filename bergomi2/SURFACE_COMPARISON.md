# How volatility surfaces differ across models

This note explains, in complete sentences, what each model can and cannot do to
an implied-volatility (IV) surface. The accompanying plots in `artifacts/` are
stylized illustrations of those capabilities; they are not PINN outputs, but
they match the qualitative shapes that Monte Carlo prices of each model produce
when inverted to Black–Scholes IV.

## Black–Scholes (constant volatility)

In Black–Scholes the only volatility input is a single number \(\sigma\).
Because that number does not depend on strike or maturity, the implied-vol
surface is a flat sheet: every option, regardless of moneyness or expiry, is
priced with the same volatility. There is therefore no smile (no U-shaped lift
in the wings away from at-the-money), no skew (no tilt that makes low-strike
puts systematically more expensive than high-strike calls), and no term
structure (short-dated and long-dated ATM vols are identical). Empirically this
is wrong—listed equity and index options show pronounced short-dated smiles and
a persistent put skew—but the flat surface remains the analytical baseline that
our PINNs correct with a learned residual \(U\).

## Heston (one stochastic variance factor)

Heston replaces constant variance with a mean-reverting CIR process for
instantaneous variance. That single factor is already enough to generate a
nontrivial smile and, once a negative spot–variance correlation is switched on,
an equity-style skew that decays as maturity grows. The fundamental limitation
is timescale: skew, smile convexity, and ATM level all relax at rates controlled
by the same mean-reversion speed \(\kappa\). You cannot make the one-month skew
very steep while leaving the two-year skew almost intact without compromise,
because there is only one clock in the dynamics. The Heston surface is therefore
richer than Black–Scholes, but its term-structure flexibility is still
one-dimensional.

## 1-factor Bergomi (one OU forward-variance factor)

The classical one-factor Bergomi model drives log forward variance with a single
Ornstein–Uhlenbeck factor \(X\). Instantaneous variance is lognormal rather than
CIR, which typically produces stronger short-dated wings than Heston for a
comparable vol-of-vol. With a moderate spot–factor correlation you see a genuine
U-shaped smile near expiry, gently tilted by skew. Dimensionally, however, the
model remains one-factor: smile, skew, and ATM term structure are all tied to the
same OU mean reversion \(\kappa\). As maturity increases, the short-dated smile
fades at that single rate, much as in Heston. Relative to Heston the surface can
look more “smiley” at short tenors; relative to two-factor Bergomi it still cannot
independently shape the short end and the long end of the surface.

## 2-factor Bergomi (fast + slow OU factors)

Two-factor Bergomi introduces a fast factor \((\omega_1,\kappa_1,\rho_1)\) and a
slow factor \((\omega_2,\kappa_2,\rho_2)\). The fast factor dominates short-dated
smile and skew—exactly the region where markets need a deep ATM trough and fat
wings—while the slow factor keeps longer ATM levels and residual skew alive after
the fast mode has mean-reverted away. That separation is the qualitative jump in
richness: the IV surface can be steep and smile-shaped at
\(\tau \approx 1\,\mathrm{M}\) and simultaneously milder but still structured at
\(\tau \approx 2\,\mathrm{Y}\), without forcing a single \(\kappa\) to do both
jobs. The plots `vol_surface_bs_vs_bergomi2.png` and
`vol_smile_bergomi2_short.png` show this as a clear U-shaped trough along the
moneyness axis that flattens gradually with maturity, set against the perfectly
flat Black–Scholes sheet. Mild correlations are deliberate in those figures so
that the usual smile is not swallowed by pure skew.

## Side-by-side differences in words

Moving from Black–Scholes to Heston is the first enrichment: a flat sheet
becomes a surface with moneyness curvature and a single maturity clock for how
that curvature dies out. Moving from Heston to one-factor Bergomi mainly changes
the shape of the short wings—lognormal variance puts more mass into deep OTM
implied vols—while leaving the “one clock” limitation intact. Moving from
one-factor to two-factor Bergomi is the step that actually adds a new degree of
freedom: short and long ends of the smile/skew term structure can be set
separately, which is why the two-factor surface looks both more convex at short
tenors and more persistent at long tenors than any one-factor competitor.

## Summary of richness

| Model | Smile | Skew | Independent short/long structure |
|-------|-------|------|-----------------------------------|
| Black–Scholes | none | none | none |
| Heston | mild | yes (one clock) | no |
| 1-factor Bergomi | stronger short wings | yes (one clock) | no |
| 2-factor Bergomi | strong short smile | yes | **yes** (two clocks) |

In the PINN stack this matters because the network only has to learn the
correction \(U\) on top of Black–Scholes: the more structured the true Bergomi
surface, the more the correction must encode those two timescales, which is why
the 2-factor feature map carries both mean-reversion clocks and both factors
explicitly.
