from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal

from jaxpensive._base import GroupSequentialBounds

MethodType = Literal["obrien_fleming", "pocock"]


class CanonicalBounds(GroupSequentialBounds):
    """Canonical group sequential stopping boundaries.

    Supports O'Brien-Fleming and Pocock methods. Both share the same
    multivariate normal / Brent's-method solver; they differ only in
    how per-look bounds are derived from the solved critical value ``c``.

    Parameters
    ----------
    reads : int
        Number of planned interim looks (>= 2).
    alpha : float
        Overall type I error rate, in (0, 1).
    sides : int
        1 for one-sided test, 2 for two-sided test.
    method : {'obrien_fleming', 'pocock'}
        Boundary shape. Must be specified explicitly.

    Examples
    --------
    >>> bounds = CanonicalBounds(reads=4, alpha=0.05, sides=2, method='obrien_fleming')
    >>> bounds.calculate_bounds()
    array([4.049..., 2.863..., 2.338..., 2.024...])

    >>> bounds = CanonicalBounds(reads=4, alpha=0.05, sides=2, method='pocock')
    >>> bounds.calculate_bounds()
    array([2.361..., 2.361..., 2.361..., 2.361...])
    """

    METHODS: tuple[str, ...] = ("obrien_fleming", "pocock")
    _MAX_Z: float = 8.0

    def __init__(
        self,
        reads: int,
        alpha: float,
        sides: int,
        method: MethodType,
        info_fractions: list[float] | None = None,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(
                f"method must be 'obrien_fleming' or 'pocock', got {method!r}"
            )
        super().__init__(reads, alpha, sides, info_fractions=info_fractions)
        self.method = method

    def _bounds_from_critical_value(self, c: float) -> np.ndarray:
        t = np.asarray(self.info_fractions, dtype=float)
        if self.method == "obrien_fleming":
            return c / np.sqrt(t)
        else:  # pocock
            return np.full(len(t), c)

    def _objective(self, c: float, cov: np.ndarray) -> float:
        """Objective for Brent's method: P(never cross boundary) - target."""
        bounds = self._bounds_from_critical_value(c)
        prob_no_cross = multivariate_normal.cdf(
            x=bounds, mean=np.zeros(self.reads), cov=cov
        )
        target = 1.0 - (self.alpha if self.sides == 1 else self.alpha / 2.0)
        return float(prob_no_cross) - target

    def calculate_bounds(self) -> np.ndarray:
        """Calculate stopping boundaries for each interim look.

        Returns
        -------
        np.ndarray
            Array of shape (K,) with critical z-values at each look.
        """
        cov = self._covariance_matrix()
        c = brentq(self._objective, 0, self._MAX_Z, args=(cov,), xtol=1e-6)
        return self._bounds_from_critical_value(c)
