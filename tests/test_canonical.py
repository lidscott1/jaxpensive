"""
Numerical tests for CanonicalBounds against published tables.

Reference values:
  O'Brien & Fleming (1979), Biometrika
  Pocock (1977), Biometrika
"""

import numpy as np
import pytest

from jaxpensive import CanonicalBounds


# ---------------------------------------------------------------------------
# O'Brien-Fleming
# ---------------------------------------------------------------------------

class TestOBrienFleming:
    def test_bounds_k4_two_sided(self):
        # Published: O'Brien & Fleming K=4, alpha=0.05, two-sided
        expected = [4.0493, 2.8627, 2.3378, 2.0245]
        bounds = CanonicalBounds(4, 0.05, 2, "obrien_fleming").calculate_bounds()
        np.testing.assert_allclose(bounds, expected, atol=1e-3)

    def test_bounds_decrease_over_time(self):
        bounds = CanonicalBounds(5, 0.05, 2, "obrien_fleming").calculate_bounds()
        assert all(bounds[i] > bounds[i + 1] for i in range(len(bounds) - 1))

    def test_one_sided_tighter_than_two_sided(self):
        b1 = CanonicalBounds(4, 0.05, 1, "obrien_fleming").calculate_bounds()
        b2 = CanonicalBounds(4, 0.05, 2, "obrien_fleming").calculate_bounds()
        assert all(b1 < b2)


# ---------------------------------------------------------------------------
# Pocock
# ---------------------------------------------------------------------------

class TestPocock:
    def test_bounds_k4_two_sided(self):
        # Published: Pocock K=4, alpha=0.05, two-sided
        expected = [2.361, 2.361, 2.361, 2.361]
        bounds = CanonicalBounds(4, 0.05, 2, "pocock").calculate_bounds()
        np.testing.assert_allclose(bounds, expected, atol=1e-3)

    def test_bounds_are_constant(self):
        bounds = CanonicalBounds(5, 0.05, 2, "pocock").calculate_bounds()
        np.testing.assert_allclose(bounds, bounds[0], rtol=1e-6)


# ---------------------------------------------------------------------------
# summary() output
# ---------------------------------------------------------------------------

class TestSummary:
    def test_two_sided_shows_both_boundaries(self, capsys):
        CanonicalBounds(4, 0.05, 2, "obrien_fleming").summary()
        out = capsys.readouterr().out
        assert "Lower (z)" in out
        assert "Upper (z)" in out
        assert "Boundary (z)" not in out

    def test_two_sided_lower_is_negative_of_upper(self, capsys):
        CanonicalBounds(4, 0.05, 2, "obrien_fleming").summary()
        out = capsys.readouterr().out
        import re
        values = [float(v) for v in re.findall(r"-?\d+\.\d+", out)]
        bounds = CanonicalBounds(4, 0.05, 2, "obrien_fleming").calculate_bounds()
        for b in bounds:
            assert pytest.approx(-b, abs=1e-3) in values
            assert pytest.approx(b, abs=1e-3) in values

    def test_one_sided_shows_single_boundary(self, capsys):
        CanonicalBounds(4, 0.05, 1, "obrien_fleming").summary()
        out = capsys.readouterr().out
        assert "Boundary (z)" in out
        assert "Lower (z)" not in out
        assert "Upper (z)" not in out


# ---------------------------------------------------------------------------
# info_fractions parameter
# ---------------------------------------------------------------------------

class TestInfoFractions:
    def test_custom_fractions_override_equally_spaced(self):
        fractions = [0.2, 0.4, 0.7, 1.0]
        b = CanonicalBounds(4, 0.05, 2, "obrien_fleming", info_fractions=fractions)
        assert b.info_fractions == fractions

    def test_equally_spaced_default_unchanged(self):
        b = CanonicalBounds(4, 0.05, 2, "obrien_fleming")
        assert b.info_fractions == [0.25, 0.5, 0.75, 1.0]

    def test_custom_fractions_change_bounds(self):
        equal = CanonicalBounds(4, 0.05, 2, "obrien_fleming")
        unequal = CanonicalBounds(4, 0.05, 2, "obrien_fleming",
                                  info_fractions=[0.1, 0.3, 0.6, 1.0])
        import numpy as np
        assert not np.allclose(equal.calculate_bounds(), unequal.calculate_bounds())

    @pytest.mark.parametrize("fractions,match", [
        ([0.25, 0.5, 0.75],        "length"),        # wrong length
        ([0.25, 0.5, 0.75, 0.9],   "1.0"),           # last != 1.0
        ([0.0, 0.5, 0.75, 1.0],    r"\(0, 1\]"),     # zero not in (0,1]
        ([0.5, 0.25, 0.75, 1.0],   "increasing"),    # not strictly increasing
        ([0.25, 0.25, 0.75, 1.0],  "increasing"),    # duplicate
    ])
    def test_invalid_info_fractions(self, fractions, match):
        with pytest.raises(ValueError, match=match):
            CanonicalBounds(4, 0.05, 2, "obrien_fleming", info_fractions=fractions)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("reads", [1, 0, -1, 1.5, "4"])
    def test_invalid_reads(self, reads):
        with pytest.raises(ValueError, match="reads"):
            CanonicalBounds(reads, 0.05, 2, "obrien_fleming")

    @pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1])
    def test_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            CanonicalBounds(4, alpha, 2, "obrien_fleming")

    @pytest.mark.parametrize("sides", [0, 3, -1, "2"])
    def test_invalid_sides(self, sides):
        with pytest.raises(ValueError, match="sides"):
            CanonicalBounds(4, 0.05, sides, "obrien_fleming")

    @pytest.mark.parametrize("method", ["POCOCK", "wang_tsiatis", "", None])
    def test_invalid_method(self, method):
        with pytest.raises((ValueError, TypeError)):
            CanonicalBounds(4, 0.05, 2, method)
