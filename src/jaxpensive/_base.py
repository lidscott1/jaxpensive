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
    info_fractions : list[float] | None, optional
        Information fractions at each look, values in (0, 1] strictly
        increasing with last value equal to 1.0. When provided, overrides
        the default equally-spaced fractions ``[1/K, 2/K, ..., 1]``.
        Length must equal ``reads``.
    """

    def __init__(
        self,
        reads: int,
        alpha: float,
        sides: int,
        info_fractions: list[float] | None = None,
    ) -> None:
        self._validate(reads, alpha, sides)
        self.reads = reads
        self.alpha = alpha
        self.sides = sides
        if info_fractions is not None:
            self._validate_info_fractions(info_fractions, reads)
            self.info_fractions: list[float] = list(info_fractions)
        else:
            self.info_fractions = [k / reads for k in range(1, reads + 1)]

    @staticmethod
    def _validate(reads: int, alpha: float, sides: int) -> None:
        if not isinstance(reads, int) or reads < 2:
            raise ValueError(f"reads must be an integer >= 2, got {reads!r}")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if sides not in (1, 2):
            raise ValueError(f"sides must be 1 or 2, got {sides!r}")

    @staticmethod
    def _validate_info_fractions(info_fractions: list[float], reads: int) -> None:
        if len(info_fractions) != reads:
            raise ValueError(
                f"info_fractions length ({len(info_fractions)}) must equal reads ({reads})"
            )
        if not all(0 < t <= 1 for t in info_fractions):
            raise ValueError("all info_fractions must be in (0, 1]")
        if not all(info_fractions[i] < info_fractions[i + 1] for i in range(len(info_fractions) - 1)):
            raise ValueError("info_fractions must be strictly increasing")
        if info_fractions[-1] != 1.0:
            raise ValueError(f"last info_fraction must be 1.0, got {info_fractions[-1]!r}")

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
