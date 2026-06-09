# jaxpensive — Claude Context

## What this package does
Group sequential testing: frequentist hypothesis testing with planned interim analyses. At each interim "look", a test statistic is compared to a stopping boundary. If it exceeds the boundary, the trial stops early. The boundaries are calibrated so the family-wise type I error rate equals `alpha` exactly.

## Domain vocabulary
- **K** — number of interim looks (called `reads` in the API)
- **t_k** — information fraction at look k: `k/K` for equally spaced looks (always in (0, 1])
- **alpha** — overall type I error rate (e.g. 0.05)
- **sides** — 1 (one-sided) or 2 (two-sided test)
- **critical value `c`** — the root found by Brent's method; bounds are derived from it
- **bounds** — the per-look stopping thresholds in z-score units

## Package structure
```
src/jaxpensive/
├── __init__.py              # public API: CanonicalBounds, AlphaSpendingBounds
├── bounds/
│   ├── _base.py             # abstract GroupSequentialBounds (summary, covariance, info_fractions)
│   └── canonical.py         # CanonicalBounds — O'Brien-Fleming and Pocock
└── spending/
    ├── __init__.py
    └── bounds.py            # AlphaSpendingBounds — Jennison-Turnbull density propagation
tests/
├── test_canonical.py
└── test_spending_bounds.py
```

## The two canonical methods
Both use the same covariance structure and Brent's root finder. They differ only in how bounds are derived from the solved critical value `c`:

| Method | Bounds formula | Character |
|--------|---------------|-----------|
| `obrien_fleming` | `c / sqrt(t_k)` | Conservative early, liberal late |
| `pocock` | `c` (constant) | Constant threshold at every look |

## Covariance matrix
The test statistics at each look follow a multivariate normal with covariance:
`Cov(Z_i, Z_j) = sqrt(min(t_i, t_j) / max(t_i, t_j))`

This is the Brownian motion covariance — a fundamental property of sequential statistics.

## Solver
`scipy.optimize.brentq` finds `c` such that the probability of never crossing the boundary under H0 equals `1 - alpha` (one-sided) or `1 - alpha/2` (two-sided). Uses `scipy.stats.multivariate_normal.cdf`.

## AlphaSpendingBounds
Uses the Jennison-Turnbull density propagation algorithm with **Gauss-Legendre quadrature**
(O(K×N²), default N=50 GL nodes). Maintains a 1D probability density over the test statistic;
at each look a `CubicSpline` is fitted to the density at the GL nodes so that both the tail
integral (inside Brent's) and the propagation integral can evaluate the density at arbitrary
points. The propagation integrates over the continuation region `[−c_k, c_k]` (two-sided) or
`[−MAX_Z, c_k]` (one-sided), keeping the kink at the stopping boundary outside the integration
domain — this is why GL achieves exponential convergence here. Accuracy vs `ldbounds` (R):
< 0.001 for Pocock and Power methods with N=50 nodes. `n_nodes` controls accuracy/speed.

Spending shapes via `method`:

| Method | Spending formula α*(t) | Character |
|--------|----------------------|-----------|
| `obf` | `2(1 − Φ(z_{α/2}/√t))` | Approximate OBF shape |
| `pocock` | `α · ln(1 + (e−1)·t)` | Approximate Pocock shape |
| `power` | `α · t^ρ` (requires `rho > 0`) | Tunable; ρ=1 is linear |

`info_fractions` on the base class enables unequally-spaced looks (unique to spending approach).

## Conventions
- All public method signatures have type hints
- Docstrings use NumPy style
- Input validation raises `ValueError` with descriptive messages at `__init__` time
- `reads` must be an integer >= 2
- `alpha` must be a float in (0, 1)
- `sides` must be 1 or 2
- `method` must be one of the class's `METHODS` tuple
- `AlphaSpendingBounds` uses `scipy.special.roots_legendre` + `scipy.interpolate.CubicSpline`; do not replace with a uniform-grid trapezoid approach

## Testing philosophy
- Tests verify numerical correctness against published tables (Pocock 1977, O'Brien & Fleming 1979)
- Never mock scipy functions — the math must actually work
- Tolerances: `atol=1e-3` is acceptable for matching published tables (which are rounded)
- Also test input validation: each invalid input should raise `ValueError`

## Published reference values for tests
O'Brien-Fleming, K=4, alpha=0.05, two-sided: bounds ≈ [4.049, 2.863, 2.338, 2.024]
Pocock, K=4, alpha=0.05, two-sided: bounds ≈ [2.361, 2.361, 2.361, 2.361]

## Status
- [x] `CanonicalBounds` with O'Brien-Fleming and Pocock — complete, 33 tests passing
- [x] Input validation, NumPy docstrings, type hints
- [x] GitHub Actions CI, BSD 3-Clause license
- [x] Demo notebook (`notebooks/demo.ipynb`)
- [x] Issue #3: `summary()` shows lower (−b_k) and upper (+b_k) columns for two-sided tests
- [x] Issue #8: `info_fractions` parameter on `GroupSequentialBounds` for unequally-spaced looks
- [x] Issue #5: `AlphaSpendingBounds` (Jennison-Turnbull) — obf, pocock, power spending — 33 tests passing
- [ ] Issue #7: demo notebook update for `AlphaSpendingBounds`
