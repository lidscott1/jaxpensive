from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from jaxpensive.bounds._base import GroupSequentialBounds

MethodType = Literal["obf", "pocock", "power"]


class AlphaSpendingBounds(GroupSequentialBounds):
    """Alpha spending group sequential stopping boundaries (Lan-DeMets).

    Uses the Jennison-Turnbull density propagation algorithm: a 1D probability
    density over the test statistic is maintained and propagated forward at each
    look via the Brownian increment transition kernel. Per-look boundaries are
    found by Brent's method on the tail integral of that density.

    Supports three spending functions, selected via ``method``:

    * ``'obf'`` — O'Brien-Fleming shape: ``α*(t) = 2(1 − Φ(z_{α/2} / √t))``
    * ``'pocock'`` — Pocock shape: ``α*(t) = α · ln(1 + (e−1)·t)``
    * ``'power'`` — Power family: ``α*(t) = α · t^ρ`` (requires ``rho > 0``)

    Parameters
    ----------
    reads : int
        Number of planned interim looks (>= 2).
    alpha : float
        Overall type I error rate, in (0, 1).
    sides : int
        1 for one-sided test, 2 for two-sided test.
    method : {'obf', 'pocock', 'power'}
        Alpha spending function shape. Must be specified explicitly.
    info_fractions : list[float] | None, optional
        Information fractions at each look. Defaults to equally spaced
        ``[1/K, 2/K, ..., 1]``. When provided, enables unequally-spaced looks.
    rho : float, optional
        Shape parameter for ``method='power'``. Ignored for other methods.
        Must be > 0. Default 1.0 (linear spending).
    n_grid : int, optional
        Number of grid points for density propagation. Higher values give
        more accurate boundaries at the cost of O(n_grid²) work per look.
        Default 200.

    Examples
    --------
    >>> b = AlphaSpendingBounds(reads=4, alpha=0.05, sides=2, method='obf')
    >>> b.calculate_bounds()
    array([4.0..., 2.8..., 2.3..., 2.0...])

    >>> b = AlphaSpendingBounds(reads=4, alpha=0.05, sides=2, method='pocock')
    >>> b.calculate_bounds()
    array([2.3..., 2.3..., 2.3..., 2.3...])
    """

    METHODS: tuple[str, ...] = ("obf", "pocock", "power")
    _MAX_Z: float = 8.0

    def __init__(
        self,
        reads: int,
        alpha: float,
        sides: int,
        method: MethodType,
        info_fractions: list[float] | None = None,
        rho: float = 1.0,
        n_grid: int = 200,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(
                f"method must be one of {self.METHODS}, got {method!r}"
            )
        if method == "power" and rho <= 0:
            raise ValueError(f"rho must be > 0 for power spending, got {rho!r}")
        super().__init__(reads, alpha, sides, info_fractions=info_fractions)
        self.method = method
        self.rho = rho
        self.n_grid = n_grid

    def _cumulative_spend(self, t: float) -> float:
        """Total cumulative alpha spent up to information time t."""
        if t <= 0.0:
            return 0.0
        if self.method == "obf":
            z = norm.ppf(1.0 - self.alpha / 2.0)
            return float(min(2.0 * (1.0 - norm.cdf(z / np.sqrt(t))), self.alpha))
        elif self.method == "pocock":
            return float(self.alpha * np.log(1.0 + (np.e - 1.0) * t))
        else:  # power
            return float(self.alpha * (t**self.rho))

    def calculate_bounds(self) -> np.ndarray:
        """Calculate stopping boundaries via Jennison-Turnbull density propagation.

        Returns
        -------
        np.ndarray
            Array of shape (K,) with critical z-values at each look.
        """
        K = self.reads
        t = np.array(self.info_fractions, dtype=float)

        grid = np.linspace(-self._MAX_Z, self._MAX_Z, self.n_grid)
        density = norm.pdf(grid)
        bounds = np.zeros(K)
        multiplier = 1.0 if self.sides == 1 else 2.0

        for k in range(K):
            t_k = float(t[k])
            t_prev = float(t[k - 1]) if k > 0 else 0.0
            alpha_k = self._cumulative_spend(t_k) - self._cumulative_spend(t_prev)

            def _tail(c: float, _d: np.ndarray = density) -> float:
                return float(np.trapezoid(np.where(grid >= c, _d, 0.0), grid))

            c_k = brentq(
                lambda c: multiplier * _tail(c) - alpha_k,
                0.0,
                self._MAX_Z,
                xtol=1e-6,
            )
            bounds[k] = c_k

            if self.sides == 1:
                density = np.where(grid <= c_k, density, 0.0)
            else:
                density = np.where(np.abs(grid) <= c_k, density, 0.0)

            if k < K - 1:
                rho_k = np.sqrt(t_k / float(t[k + 1]))
                sigma_k = np.sqrt(1.0 - rho_k**2)
                kernel = norm.pdf(
                    (grid[np.newaxis, :] - rho_k * grid[:, np.newaxis]) / sigma_k
                ) / sigma_k
                density = np.trapezoid(density[:, np.newaxis] * kernel, grid, axis=0)

        return bounds

    def __repr__(self) -> str:
        rho_part = f", rho={self.rho!r}" if self.method == "power" else ""
        return (
            f"{self.__class__.__name__}("
            f"reads={self.reads}, alpha={self.alpha}, sides={self.sides}, "
            f"method={self.method!r}{rho_part})"
        )
