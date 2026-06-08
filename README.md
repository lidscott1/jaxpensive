# jaxpensive

Group sequential testing bounds for interim analyses in clinical trials and experiments.

## Installation

```bash
pip install jaxpensive
```

## Usage

```python
from jaxpensive import CanonicalBounds

# O'Brien-Fleming bounds, 4 looks, two-sided alpha=0.05
bounds = CanonicalBounds(reads=4, alpha=0.05, sides=2, method="obrien_fleming")
bounds.summary()

# Pocock bounds
bounds = CanonicalBounds(reads=4, alpha=0.05, sides=2, method="pocock")
bounds.summary()
```
