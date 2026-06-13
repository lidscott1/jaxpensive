# jaxpensive

jaxpensive calculates group sequential testing bounds for interim analyses for statistical experiments.  It is a python implementation and attempts to mirror calculations in more mature R libraries like `gsDesign` and `ldbounds`.  

This package was written because to the authors knowledge there is no public sequential bounds calculator given as a pure python implementation.  As this package is in its infancy if the readers intention is to use this package for anything of material importance (i.e. clinical trials) it is recommended that you verify the values given by the boundary calculation using one of the R libraries, or checking the values against the multivariate normal specified under the null hypothesis.  

### Technical limitations

At present this is package assumes symmetric bounds for 2 sided tests.  In future enhancements non symmetric bounds may be implemented.  But if the readers requirement is to implement them now, it is not available and should be calculated with one of the packages above.

## Boundary calculators

jaxpensive provides 2 types of boundary calculators, a canonical implementation and a spending function implementation.  

### Canonical implementation

The canonical implementation creates bounds using direct multivariate normal integration.  As such it is easy to implement.  However, it is computationally expensive to calculate.   As a rule of thumb, if your number of reads grows beyond 10 computations will slow considerably.

### Spending function implementation

This solves sequential boundaries exploiting properties of brownian motion that mirror how a z-score behaves under the null hypothesis.  Using the jennison turnbull algorithm, it exploits a brownian motion transition to recursively calculate a series of sequential single dimension densities replacing expensive multivariate normal integration.  This allows for much speedier calculation and accomodates boundaries with reads much larger than 10.

## Installation

```bash
pip install jaxpensive
```

## Usage

### Canonical bounds

```python
from jaxpensive import CanonicalBounds

# O'Brien-Fleming bounds, 4 looks, two-sided alpha=0.05
bounds = CanonicalBounds(reads=4, alpha=0.05, sides=2, method="obrien_fleming")
bounds.summary()

# Pocock bounds
bounds = CanonicalBounds(reads=4, alpha=0.05, sides=2, method="pocock")
bounds.summary()
```

### Spending function bounds

```python
from jaxpensive import AlphaSpendingBounds

# O'Brien-Fleming spending shape
bounds = AlphaSpendingBounds(reads=4, alpha=0.05, sides=2, method="obf")
bounds.summary()

# Pocock spending shape
bounds = AlphaSpendingBounds(reads=4, alpha=0.05, sides=2, method="pocock")
bounds.summary()

# Power family (rho controls aggressiveness of early spending;
# rho < 1 spends more early, rho > 1 spends more late)
bounds = AlphaSpendingBounds(reads=4, alpha=0.05, sides=2, method="power", rho=0.5)
bounds.summary()
```

#### Unequally-spaced looks

Unlike canonical bounds, the spending function approach supports looks that are not equally spaced in information time:

```python
bounds = AlphaSpendingBounds(
    reads=4, alpha=0.05, sides=2, method="obf",
    info_fractions=[0.2, 0.4, 0.7, 1.0],
)
bounds.summary()
```

#### Custom spending functions

Any spending function can be supplied directly. See [Custom spending function requirements](#custom-spending-function-requirements) below for the full specification.

```python
import numpy as np
from scipy.stats import norm

# Hwang-Shih-DeCani family (phi < 0: OBF-like, phi > 0: back-loaded)
alpha = 0.05
phi = -4.0

def hwang_shih_decani(t):
    return alpha * (1 - np.exp(-phi * t)) / (1 - np.exp(-phi))

bounds = AlphaSpendingBounds(
    reads=4, alpha=alpha, sides=2, method="custom",
    spending_fn=hwang_shih_decani,
)
bounds.summary()
```

## Custom spending function requirements

A valid spending function `f(t)` represents the cumulative type I error spent by information time `t`. It must satisfy the following:

| Requirement | Details |
|---|---|
| **Signature** | `f(t: float) -> float` |
| **Zero at origin** | `f(0) = 0` — no alpha is spent before any data is seen |
| **Totals to alpha** | `f(1) = alpha` — all alpha is spent by the final look (tolerance `1e-4`) |
| **Non-decreasing** | `f` should be monotonically non-decreasing on `(0, 1]` — alpha once spent cannot be reclaimed |
| **Range** | Return values are clipped to `[0, alpha]` internally, but the function should not return values outside this range |

The first two conditions (`f(0) = 0` and `f(1) ≈ alpha`) are validated at construction time and will raise a `ValueError` if violated. Non-monotonicity is not validated but will produce nonsensical or negative incremental spend at individual looks, leading to a failed boundary solve.

The function receives a single argument `t` in `(0, 1]` and should be defined for all values in that interval. It does not receive `alpha` as an argument — capture it from the enclosing scope:

```python
alpha = 0.05

def my_spending_fn(t: float) -> float:
    # alpha is captured from the enclosing scope
    return alpha * t ** 0.75

bounds = AlphaSpendingBounds(reads=4, alpha=alpha, sides=2, method="custom",
                              spending_fn=my_spending_fn)
```
