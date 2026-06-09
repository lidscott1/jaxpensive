from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import roots_legendre
from scipy.stats import norm

from jaxpensive.bounds._base import GroupSequentialBounds

MethodType = Literal["obf", "pocock", "power"]


class AlphaSpendingBounds(GroupSequentialBounds):
    """Alpha spending group sequential stopping boundaries (Lan-DeMets).

    Uses the Jennison-Turnbull density propagation algorithm with Gauss-Legendre
    quadrature. A 1D probability density over the test statistic is maintained
    at GL nodes and propagated look-to-look via the Brownian increment transition
    kernel. At each step the integration is performed over the continuation region
    only, so the kink at the stopping boundary never contaminates the quadrature.
    A cubic spline built on the GL nodes lets the tail-integral (used inside
    Brent's method) and the propagation integral both evaluate the density
    accurately at arbitrary points within that region.

    Compared to a uniform-grid trapezoid approach, GL quadrature achieves
    exponential convergence for smooth integrands and requires far fewer nodes
    for equivalent accuracy (default ``n_nodes=50`` vs ~200+ for trapezoid).

    Supports three spending functions, selected via ``method``:

    * ``'obf'``    — O'Brien-Fleming shape: ``α*(t) = 2(1 − Φ(z_{α/2} / √t))``
    * ``'pocock'`` — Pocock shape: ``α*(t) = α · ln(1 + (e−1)·t)``
    * ``'power'``  — Power family: ``α*(t) = α · t^ρ`` (requires ``rho > 0``)

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
    n_nodes : int, optional
        Number of Gauss-Legendre nodes for density representation and
        integration. Cost is O(n_nodes²) per look. Default 50.

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
        n_nodes: int = 50,
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
        self.n_nodes = n_nodes

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

    @staticmethod
    def _gl_map(
        a: float, b: float, xi: np.ndarray, wi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map standard GL nodes/weights on [-1, 1] onto [a, b]."""
        scale = 0.5 * (b - a)
        return scale * xi + 0.5 * (a + b), scale * wi

    def calculate_bounds(self) -> np.ndarray:
        """Calculate stopping boundaries via Jennison-Turnbull with GL quadrature.

        Returns
        -------
        np.ndarray
            Array of shape (K,) with critical z-values at each look.
        """
        K = self.reads
        t = np.array(self.info_fractions, dtype=float)
        multiplier = 1.0 if self.sides == 1 else 2.0

        # GL nodes and weights on [-1, 1] — precomputed once, reused every look
        xi, wi = roots_legendre(self.n_nodes)

        # Fixed output nodes on [-MAX_Z, MAX_Z]: density is stored here each look
        z_out, _ = self._gl_map(-self._MAX_Z, self._MAX_Z, xi, wi)

        # Initial density: standard normal at the output nodes
        density_vals = norm.pdf(z_out)

        bounds = np.zeros(K)

        for k in range(K):
            t_k = float(t[k])
            t_prev = float(t[k - 1]) if k > 0 else 0.0
            alpha_k = self._cumulative_spend(t_k) - self._cumulative_spend(t_prev)

            # Cubic spline over the GL nodes — smooth, well-conditioned interpolant
            # for evaluating the density at arbitrary points within [-MAX_Z, MAX_Z].
            # extrapolate=False gives NaN outside the range; we replace with 0.
            cs = CubicSpline(z_out, density_vals, extrapolate=False)

            # Tail integral ∫_{c}^{MAX_Z} density dz via GL on [c, MAX_Z].
            # Called ~20 times per look during Brent's iteration.
            def _tail(c: float, _cs: CubicSpline = cs) -> float:
                if c >= self._MAX_Z:
                    return 0.0
                z_t, w_t = self._gl_map(c, self._MAX_Z, xi, wi)
                d_t = np.nan_to_num(_cs(z_t), nan=0.0)
                np.clip(d_t, 0.0, None, out=d_t)
                return float(np.dot(w_t, d_t))

            c_k = brentq(
                lambda c: multiplier * _tail(c) - alpha_k,
                0.0,
                self._MAX_Z,
                xtol=1e-6,
            )
            bounds[k] = c_k

            if k < K - 1:
                rho_k = np.sqrt(t_k / float(t[k + 1]))
                sigma_k = np.sqrt(1.0 - rho_k**2)

                # Integrate over the continuation region only — no kink at ±c_k.
                # Two-sided: [-c_k, c_k].  One-sided: [-MAX_Z, c_k].
                a_int = -self._MAX_Z if self.sides == 1 else -c_k
                z_int, w_int = self._gl_map(a_int, c_k, xi, wi)
                d_int = np.nan_to_num(cs(z_int), nan=0.0)
                np.clip(d_int, 0.0, None, out=d_int)

                # Brownian transition kernel: shape (n_out, n_int)
                # kernel[i, j] = φ((z_out[i] - rho * z_int[j]) / sigma) / sigma
                kernel = norm.pdf(
                    (z_out[:, np.newaxis] - rho_k * z_int[np.newaxis, :]) / sigma_k
                ) / sigma_k

                density_vals = kernel @ (w_int * d_int)

        return bounds

    def __repr__(self) -> str:
        rho_part = f", rho={self.rho!r}" if self.method == "power" else ""
        return (
            f"{self.__class__.__name__}("
            f"reads={self.reads}, alpha={self.alpha}, sides={self.sides}, "
            f"method={self.method!r}{rho_part})"
        )
