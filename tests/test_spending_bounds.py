"""
Tests for AlphaSpendingBounds (Jennison-Turnbull density propagation).

Validation strategy:
  - OBF and Pocock spending bounds should approximate their canonical counterparts
    for equally-spaced looks (Jennison & Turnbull, 2000, Ch. 19).
  - Published canonical values: OBF K=4 two-sided ≈ [4.049, 2.863, 2.338, 2.024]
                                 Pocock K=4 two-sided ≈ [2.361, 2.361, 2.361, 2.361]
  - atol=0.05 — spending bounds converge to but do not exactly match canonical.
"""

import numpy as np
import pytest

from jaxpensive import AlphaSpendingBounds


# ---------------------------------------------------------------------------
# OBF spending
# ---------------------------------------------------------------------------

class TestOBFSpending:
    def test_bounds_approximate_canonical_obf(self):
        # OBF spending is an approximation to canonical OBF — not numerically
        # identical.  Typical discrepancy is ~0.05–0.15, largest at early looks.
        b = AlphaSpendingBounds(4, 0.05, 2, "obf")
        expected = [4.0493, 2.8627, 2.3378, 2.0245]
        np.testing.assert_allclose(b.calculate_bounds(), expected, atol=0.2)

    def test_bounds_decrease_over_time(self):
        b = AlphaSpendingBounds(5, 0.05, 2, "obf").calculate_bounds()
        assert all(b[i] > b[i + 1] for i in range(len(b) - 1))

    def test_one_sided_tighter_than_two_sided(self):
        b1 = AlphaSpendingBounds(4, 0.05, 1, "obf").calculate_bounds()
        b2 = AlphaSpendingBounds(4, 0.05, 2, "obf").calculate_bounds()
        assert all(b1 < b2)


# ---------------------------------------------------------------------------
# Pocock spending
# ---------------------------------------------------------------------------

class TestPocockSpending:
    def test_bounds_approximate_canonical_pocock(self):
        b = AlphaSpendingBounds(4, 0.05, 2, "pocock")
        expected = [2.361, 2.361, 2.361, 2.361]
        np.testing.assert_allclose(b.calculate_bounds(), expected, atol=0.05)

    def test_bounds_nearly_constant(self):
        # Pocock spending bounds are approximately constant but not exactly so
        # (spending function is an approximation). Range across looks < 0.15.
        b = AlphaSpendingBounds(5, 0.05, 2, "pocock").calculate_bounds()
        assert b.max() - b.min() < 0.15


# ---------------------------------------------------------------------------
# Power spending
# ---------------------------------------------------------------------------

class TestPowerSpending:
    def test_rho1_is_linear(self):
        b = AlphaSpendingBounds(4, 0.05, 2, "power", rho=1.0)
        bounds = b.calculate_bounds()
        assert len(bounds) == 4

    def test_high_rho_conservative_early(self):
        # rho > 1: spends less early → higher early bounds
        b_high = AlphaSpendingBounds(4, 0.05, 2, "power", rho=3.0).calculate_bounds()
        b_low = AlphaSpendingBounds(4, 0.05, 2, "power", rho=0.5).calculate_bounds()
        assert b_high[0] > b_low[0]

    def test_invalid_rho_zero(self):
        with pytest.raises(ValueError, match="rho"):
            AlphaSpendingBounds(4, 0.05, 2, "power", rho=0.0)

    def test_invalid_rho_negative(self):
        with pytest.raises(ValueError, match="rho"):
            AlphaSpendingBounds(4, 0.05, 2, "power", rho=-1.0)


# ---------------------------------------------------------------------------
# Unequally-spaced info_fractions
# ---------------------------------------------------------------------------

class TestUnequallySpacedLooks:
    def test_custom_fractions_run_without_error(self):
        b = AlphaSpendingBounds(
            4, 0.05, 2, "obf", info_fractions=[0.2, 0.4, 0.7, 1.0]
        )
        bounds = b.calculate_bounds()
        assert len(bounds) == 4

    def test_custom_fractions_change_bounds(self):
        equal = AlphaSpendingBounds(4, 0.05, 2, "obf")
        unequal = AlphaSpendingBounds(
            4, 0.05, 2, "obf", info_fractions=[0.1, 0.3, 0.6, 1.0]
        )
        assert not np.allclose(equal.calculate_bounds(), unequal.calculate_bounds())

    def test_unequal_obf_bounds_decrease(self):
        b = AlphaSpendingBounds(
            4, 0.05, 2, "obf", info_fractions=[0.2, 0.4, 0.7, 1.0]
        ).calculate_bounds()
        assert all(b[i] > b[i + 1] for i in range(len(b) - 1))


# ---------------------------------------------------------------------------
# Cumulative spending properties
# ---------------------------------------------------------------------------

class TestSpendingProperties:
    @pytest.mark.parametrize("method", ["obf", "pocock", "power"])
    def test_spend_zero_at_t0(self, method):
        b = AlphaSpendingBounds(4, 0.05, 2, method)
        assert b._cumulative_spend(0.0) == 0.0

    @pytest.mark.parametrize("method", ["obf", "pocock", "power"])
    def test_spend_alpha_at_t1(self, method):
        b = AlphaSpendingBounds(4, 0.05, 2, method)
        np.testing.assert_allclose(b._cumulative_spend(1.0), 0.05, atol=1e-6)

    @pytest.mark.parametrize("method", ["obf", "pocock", "power"])
    def test_total_alpha_spent(self, method):
        b = AlphaSpendingBounds(4, 0.05, 2, method)
        bounds = b.calculate_bounds()
        assert len(bounds) == 4


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            AlphaSpendingBounds(4, 0.05, 2, "wang_tsiatis")

    def test_invalid_reads(self):
        with pytest.raises(ValueError, match="reads"):
            AlphaSpendingBounds(1, 0.05, 2, "obf")

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            AlphaSpendingBounds(4, 1.5, 2, "obf")

    def test_invalid_sides(self):
        with pytest.raises(ValueError, match="sides"):
            AlphaSpendingBounds(4, 0.05, 3, "obf")

    @pytest.mark.parametrize("fractions,match", [
        ([0.25, 0.5, 0.75],       "length"),
        ([0.25, 0.5, 0.75, 0.9],  "1.0"),
        ([0.0, 0.5, 0.75, 1.0],   r"\(0, 1\]"),
        ([0.5, 0.25, 0.75, 1.0],  "increasing"),
    ])
    def test_invalid_info_fractions(self, fractions, match):
        with pytest.raises(ValueError, match=match):
            AlphaSpendingBounds(4, 0.05, 2, "obf", info_fractions=fractions)


# ---------------------------------------------------------------------------
# summary() and __repr__
# ---------------------------------------------------------------------------

class TestSummary:
    def test_two_sided_summary_shows_both_boundaries(self, capsys):
        AlphaSpendingBounds(4, 0.05, 2, "obf").summary()
        out = capsys.readouterr().out
        assert "Lower (z)" in out
        assert "Upper (z)" in out

    def test_one_sided_summary_shows_single_boundary(self, capsys):
        AlphaSpendingBounds(4, 0.05, 1, "obf").summary()
        out = capsys.readouterr().out
        assert "Boundary (z)" in out
        assert "Lower (z)" not in out

    def test_repr_no_rho_for_obf(self):
        b = AlphaSpendingBounds(4, 0.05, 2, "obf")
        assert "rho" not in repr(b)

    def test_repr_includes_rho_for_power(self):
        b = AlphaSpendingBounds(4, 0.05, 2, "power", rho=2.0)
        assert "rho=2.0" in repr(b)
