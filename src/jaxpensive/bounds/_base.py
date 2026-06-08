from abc import ABC, abstractmethod

import numpy as np


class GroupSequentialBounds(ABC):
    """Abstract base class for group sequential stopping boundaries.

    Parameters
    ----------
    reads : int
        Number of planned interim looks (>= 2).
    alpha : float
        Overall type I error rate, in (0, 1).
    sides : int
        1 for one-sided test, 2 for two-sided test.
    """

    def __init__(self, reads: int, alpha: float, sides: int) -> None:
        self._validate(reads, alpha, sides)
        self.reads = reads
        self.alpha = alpha
        self.sides = sides
        self.info_fractions: list[float] = [k / reads for k in range(1, reads + 1)]

    @staticmethod
    def _validate(reads: int, alpha: float, sides: int) -> None:
        if not isinstance(reads, int) or reads < 2:
            raise ValueError(f"reads must be an integer >= 2, got {reads!r}")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if sides not in (1, 2):
            raise ValueError(f"sides must be 1 or 2, got {sides!r}")

    def _covariance_matrix(self) -> np.ndarray:
        t = np.asarray(self.info_fractions, dtype=float)
        return np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))

    @abstractmethod
    def calculate_bounds(self) -> np.ndarray:
        """Calculate stopping boundaries for each interim look.

        Returns
        -------
        np.ndarray
            Array of shape (K,) with critical z-values at each look.
        """

    def summary(self) -> None:
        """Print a formatted table of test parameters and stopping boundaries.

        For two-sided tests both the lower (−b_k) and upper (+b_k) boundaries
        are shown. For one-sided tests a single boundary column is shown.
        """
        bounds = self.calculate_bounds()
        method_name = getattr(self, "method", self.__class__.__name__)

        if self.sides == 2:
            header = (
                f"  {'Look':>4}  {'Info fraction':>14}"
                f"  {'Lower (z)':>10}  {'Upper (z)':>10}"
            )
            rows = [
                f"  {k + 1:>4}  {t:>14.4f}  {-b:>10.4f}  {b:>10.4f}"
                for k, (t, b) in enumerate(zip(self.info_fractions, bounds))
            ]
        else:
            header = f"  {'Look':>4}  {'Info fraction':>14}  {'Boundary (z)':>12}"
            rows = [
                f"  {k + 1:>4}  {t:>14.4f}  {b:>12.4f}"
                for k, (t, b) in enumerate(zip(self.info_fractions, bounds))
            ]

        sep = "  " + "-" * (len(header) - 2)

        print(f"\nGroup Sequential Bounds — {method_name}")
        print(f"  Looks : {self.reads}")
        print(f"  Alpha : {self.alpha}")
        print(f"  Sides : {self.sides}")
        print(sep)
        print(header)
        print(sep)
        for row in rows:
            print(row)
        print(sep)

    def __repr__(self) -> str:
        method = getattr(self, "method", "")
        method_part = f", method={self.method!r}" if method else ""
        return (
            f"{self.__class__.__name__}("
            f"reads={self.reads}, alpha={self.alpha}, sides={self.sides}"
            f"{method_part})"
        )
