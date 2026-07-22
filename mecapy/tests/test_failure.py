"""Tests for the static failure criteria (Shigley Ch. 5)."""

import math

import pytest

from mecapy.failure import (
    n_coulomb_mohr,
    n_distortion_energy,
    n_maximum_shear_stress,
    n_modified_mohr,
    principal_stresses,
    von_mises,
)


class TestVonMises:
    """von_mises and principal_stresses."""

    def test_uniaxial(self):
        """A uniaxial stress equals its own von Mises value."""
        assert von_mises(sx=100.0) == pytest.approx(100.0)

    def test_pure_shear(self):
        """Pure shear gives sigma' = sqrt(3) * tau."""
        assert von_mises(txy=50.0) == pytest.approx(math.sqrt(3) * 50.0)

    def test_plane_stress_hand_calc(self):
        """von Mises of (sx=80, txy=50) matches the hand calculation."""
        assert von_mises(sx=80.0, txy=50.0) == pytest.approx(
            math.sqrt(13900.0))

    def test_principal_stresses_ordered(self):
        """Principals of (80, 0, 50) are Mohr's-circle center +/- radius, 0."""
        s1, s2, s3 = principal_stresses(sx=80.0, sy=0.0, txy=50.0)
        radius = math.sqrt(40.0 ** 2 + 50.0 ** 2)
        assert s1 == pytest.approx(40.0 + radius)
        assert s2 == pytest.approx(0.0)
        assert s3 == pytest.approx(40.0 - radius)


class TestDuctileCriteria:
    """Distortion-energy and maximum-shear-stress safety factors."""

    def test_distortion_energy_uniaxial(self):
        """DE reduces to Sy/sigma for a uniaxial stress."""
        assert n_distortion_energy(250.0, sx=100.0) == pytest.approx(2.5)

    def test_distortion_energy_zero_stress_is_infinite(self):
        """No stress means infinite safety factor."""
        assert n_distortion_energy(250.0) == math.inf

    def test_mss_pure_shear(self):
        """MSS of pure shear tau: principals (tau, 0, -tau), n = Sy/(2 tau)."""
        s1, _, s3 = principal_stresses(txy=60.0)
        assert n_maximum_shear_stress(250.0, s1, s3) == pytest.approx(
            250.0 / 120.0)

    def test_mss_more_conservative_than_de(self):
        """For pure shear MSS gives a lower factor than DE."""
        s1, _, s3 = principal_stresses(txy=60.0)
        assert n_maximum_shear_stress(250.0, s1, s3) < n_distortion_energy(
            250.0, txy=60.0)

    def test_negative_yield_raises(self):
        """A non-positive yield strength is rejected."""
        with pytest.raises(ValueError):
            n_distortion_energy(0.0, sx=10.0)
        with pytest.raises(ValueError):
            n_maximum_shear_stress(-1.0, 10.0, 0.0)

    def test_mss_requires_ordering(self):
        """sigma_1 must not be below sigma_3."""
        with pytest.raises(ValueError):
            n_maximum_shear_stress(250.0, 10.0, 50.0)


class TestBrittleCriteria:
    """Coulomb-Mohr and modified-Mohr safety factors."""

    def test_coulomb_mohr_tension(self):
        """Pure tension reduces to Sut/sigma_1."""
        assert n_coulomb_mohr(100.0, 0.0, 250.0, 820.0) == pytest.approx(2.5)

    def test_coulomb_mohr_mixed_quadrant(self):
        """Mixed tension/compression uses 1/(s1/Sut - s3/Suc)."""
        n = n_coulomb_mohr(100.0, -50.0, 250.0, 820.0)
        assert n == pytest.approx(1.0 / (100.0 / 250.0 + 50.0 / 820.0))

    def test_coulomb_mohr_compression(self):
        """Pure compression uses Suc/|sigma_3|."""
        assert n_coulomb_mohr(-50.0, -300.0, 250.0, 820.0) == pytest.approx(
            820.0 / 300.0)

    def test_modified_mohr_small_compression_uses_sut(self):
        """|sigma_B| <= sigma_A stays on the Sut/sigma_1 branch."""
        assert n_modified_mohr(100.0, -50.0, 250.0, 820.0) == pytest.approx(
            2.5)

    def test_modified_mohr_fourth_quadrant(self):
        """|sigma_B| > sigma_A uses the modified-Mohr slanted line."""
        n = n_modified_mohr(50.0, -200.0, 250.0, 820.0)
        expected = 1.0 / ((820.0 - 250.0) * 50.0 / (820.0 * 250.0)
                          + 200.0 / 820.0)
        assert n == pytest.approx(expected)

    def test_modified_mohr_less_conservative_in_fourth_quadrant(self):
        """Modified Mohr is not more conservative than Coulomb-Mohr."""
        cm = n_coulomb_mohr(50.0, -200.0, 250.0, 820.0)
        mm = n_modified_mohr(50.0, -200.0, 250.0, 820.0)
        assert mm >= cm

    def test_bad_strengths_raise(self):
        """Non-positive Sut or Suc is rejected."""
        with pytest.raises(ValueError):
            n_coulomb_mohr(100.0, 0.0, 0.0, 820.0)
        with pytest.raises(ValueError):
            n_modified_mohr(100.0, 0.0, 250.0, -1.0)
