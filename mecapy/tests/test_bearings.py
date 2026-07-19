"""Tests for bearing module."""

from math import isclose

import pytest
from mecapy.bearings import Bearing
from mecapy.bearings.bearing_data import (
    get_application_factor,
    get_xy_factors,
    weibull_life_multiplier,
    weibull_reliability,
)


def make_ball_bearing(**kwargs):
    """Standard 6205-ish ball bearing used across tests."""
    defaults = dict(
        bore_diameter=25.0,
        outer_diameter=52.0,
        width=15.0,
        bearing_type="ball",
        C10=35000.0,
        C0=14000.0,
    )
    defaults.update(kwargs)
    return Bearing(**defaults)


class TestBearing:
    """Test cases for Bearing class."""

    def test_bearing_creation(self):
        """Test creating a bearing object."""
        bearing = Bearing(
            bore_diameter=10.0,
            outer_diameter=26.0,
            width=8.0,
            bearing_type="ball",
        )
        assert bearing.bore_diameter == 10.0
        assert bearing.outer_diameter == 26.0
        assert bearing.width == 8.0
        assert bearing.bearing_type == "ball"

    def test_bearing_repr(self):
        """Test bearing string representation."""
        bearing = Bearing(
            bore_diameter=20.0,
            outer_diameter=52.0,
            width=15.0,
            bearing_type="roller",
        )
        assert "Bearing" in repr(bearing)


class TestBearingValidation:
    """Constructor validation."""

    def test_non_positive_bore_raises(self):
        """Zero or negative bore diameter is rejected."""
        with pytest.raises(ValueError):
            Bearing(bore_diameter=0, outer_diameter=26.0, width=8.0)

    def test_non_positive_width_raises(self):
        """Zero or negative width is rejected."""
        with pytest.raises(ValueError):
            Bearing(bore_diameter=10.0, outer_diameter=26.0, width=-1.0)

    def test_outer_not_larger_than_bore_raises(self):
        """Outer diameter must exceed the bore."""
        with pytest.raises(ValueError):
            Bearing(bore_diameter=26.0, outer_diameter=26.0, width=8.0)

    def test_unknown_bearing_type_raises(self):
        """Unknown bearing type is rejected with available options."""
        with pytest.raises(ValueError):
            Bearing(
                bore_diameter=10.0,
                outer_diameter=26.0,
                width=8.0,
                bearing_type="magnetic",
            )

    def test_negative_rating_raises(self):
        """Negative load ratings are rejected."""
        with pytest.raises(ValueError):
            make_ball_bearing(C10=-1.0)
        with pytest.raises(ValueError):
            make_ball_bearing(C0=0.0)


class TestBearingLife:
    """Load-life relation (eq. 11-1)."""

    def test_ball_life(self):
        """L10 = 1e6 * (C10/F)^3 for a ball bearing."""
        bearing = make_ball_bearing()
        assert isclose(bearing.life(7000.0), 125.0e6, rel_tol=1e-4)

    def test_roller_life_exponent(self):
        """Roller bearings use the 10/3 exponent."""
        bearing = make_ball_bearing(bearing_type="roller")
        assert isclose(bearing.life(7000.0), 1e6 * 5.0 ** (10.0 / 3.0), rel_tol=1e-4)

    def test_life_hours(self):
        """Life in hours divides by 60*rpm."""
        bearing = make_ball_bearing()
        assert isclose(
            bearing.life_hours(7000.0, 500.0), 125.0e6 / 30000.0, rel_tol=1e-4
        )

    def test_application_factor_shortens_life(self):
        """af scales the load before the life exponent."""
        bearing = make_ball_bearing()
        assert isclose(
            bearing.life(7000.0, application_factor=1.2), 125.0e6 / 1.2**3, rel_tol=1e-4
        )

    def test_life_without_c10_raises(self):
        """Life methods need a C10 rating."""
        bearing = Bearing(bore_diameter=10.0, outer_diameter=26.0, width=8.0)
        with pytest.raises(ValueError):
            bearing.life(1000.0)

    def test_application_factor_lookup(self):
        """Table 11-5 lookup returns the range endpoints and midpoint."""
        assert get_application_factor("light impact", "min") == 1.2
        assert get_application_factor("light impact", "max") == 1.5
        assert isclose(get_application_factor("light impact"), 1.35, rel_tol=1e-9)
        with pytest.raises(ValueError):
            get_application_factor("space launch")


class TestBearingReliability:
    """Weibull reliability model (eq. 11-5 to 11-10)."""

    def test_weibull_anchors(self):
        """Known multipliers at R = 0.90 and R = 0.99."""
        assert isclose(weibull_life_multiplier(0.90), 0.9931, rel_tol=1e-3)
        assert isclose(weibull_life_multiplier(0.99), 0.2196, rel_tol=2e-3)

    def test_reliability_round_trip(self):
        """reliability(adjusted_life(R)) returns R."""
        bearing = make_ball_bearing()
        life = bearing.adjusted_life(7000.0, reliability=0.95)
        assert isclose(bearing.reliability(7000.0, life), 0.95, rel_tol=1e-6)

    def test_required_c10_round_trip(self):
        """A bearing rated at required_C10 reaches the desired life."""
        bearing = make_ball_bearing()
        c10 = bearing.required_C10(7000.0, 50.0e6, reliability=0.99)
        sized = make_ball_bearing(C10=c10)
        assert isclose(
            sized.adjusted_life(7000.0, reliability=0.99), 50.0e6, rel_tol=1e-6
        )

    def test_approximate_close_to_exact(self):
        """Eq. 11-10 approximation agrees with the exact inversion."""
        bearing = make_ball_bearing()
        exact = bearing.required_C10(7000.0, 50.0e6, reliability=0.99)
        approx = bearing.required_C10(
            7000.0, 50.0e6, reliability=0.99, approximate=True
        )
        assert isclose(exact, approx, rel_tol=2e-2)

    def test_invalid_reliability_raises(self):
        """Reliability outside (0, 1) is rejected."""
        bearing = make_ball_bearing()
        for bad in (0.0, 1.0, 1.2, -0.1):
            with pytest.raises(ValueError):
                bearing.adjusted_life(7000.0, reliability=bad)

    def test_weibull_reliability_below_x0(self):
        """Lives below the guaranteed life have R = 1."""
        assert weibull_reliability(0.01) == 1.0


class TestEquivalentLoad:
    """Combined radial and axial loading (table 11-1, eq. 11-12)."""

    def test_combined_load_anchor(self):
        """Interpolated X/Y factors reproduce a hand calculation."""
        # Fa/C0 = 1400/14000 = 0.100 -> e ~ 0.2923, Y2 ~ 1.4885
        # Fa/(V*Fr) = 0.4667 > e -> P = 0.56*3000 + 1.4885*1400
        bearing = make_ball_bearing()
        p = bearing.equivalent_load(3000.0, 1400.0)
        assert isclose(p, 3763.9, rel_tol=1e-3)

    def test_below_e_branch(self):
        """Small axial loads leave P = V * Fr."""
        bearing = make_ball_bearing()
        assert isclose(bearing.equivalent_load(3000.0, 100.0), 3000.0, rel_tol=1e-9)

    def test_outer_ring_rotation(self):
        """Outer-ring rotation applies V = 1.2."""
        bearing = make_ball_bearing()
        assert isclose(
            bearing.equivalent_load(3000.0, rotating="outer"), 3600.0, rel_tol=1e-9
        )
        with pytest.raises(ValueError):
            bearing.equivalent_load(3000.0, rotating="cage")

    def test_roller_with_axial_raises(self):
        """Table 11-1 is for ball bearings only."""
        bearing = make_ball_bearing(bearing_type="roller")
        with pytest.raises(ValueError):
            bearing.equivalent_load(3000.0, 1400.0)
        # pure radial load is fine
        assert isclose(bearing.equivalent_load(3000.0), 3000.0, rel_tol=1e-9)

    def test_axial_without_c0_raises(self):
        """Combined loading needs the static rating C0."""
        bearing = Bearing(
            bore_diameter=25.0, outer_diameter=52.0, width=15.0, C10=35000.0
        )
        with pytest.raises(ValueError):
            bearing.equivalent_load(3000.0, 1400.0)

    def test_xy_factor_interpolation(self):
        """Table row values are recovered exactly and clamped outside."""
        e, x2, y2 = get_xy_factors(0.084)
        assert isclose(e, 0.28, rel_tol=1e-9)
        assert isclose(x2, 0.56, rel_tol=1e-9)
        assert isclose(y2, 1.55, rel_tol=1e-9)
        e_hi, _, y2_hi = get_xy_factors(1.0)  # clamped to last row
        assert isclose(e_hi, 0.44, rel_tol=1e-9)
        assert isclose(y2_hi, 1.00, rel_tol=1e-9)
        with pytest.raises(ValueError):
            get_xy_factors(-0.1)


class TestVariableLoading:
    """Duty-cycle (cubic mean) loading (sec. 11-10)."""

    def test_two_segment_cubic_mean(self):
        """Equal-time two-level loading matches the hand calculation."""
        bearing = make_ball_bearing()
        feq = bearing.equivalent_steady_load([(1000.0, 0.5), (2000.0, 0.5)])
        assert isclose(feq, (0.5 * 1e9 + 0.5 * 8e9) ** (1.0 / 3.0), rel_tol=1e-6)

    def test_speed_weighted_segments(self):
        """Speeds weight the revolutions accumulated per segment."""
        bearing = make_ball_bearing()
        feq = bearing.equivalent_steady_load(
            [(1000.0, 0.5, 300.0), (2000.0, 0.5, 900.0)]
        )
        expected = ((150.0 * 1e9 + 450.0 * 8e9) / 600.0) ** (1.0 / 3.0)
        assert isclose(feq, expected, rel_tol=1e-6)

    def test_per_segment_application_factor(self):
        """4-tuples scale each segment load by its af."""
        bearing = make_ball_bearing()
        feq = bearing.equivalent_steady_load([(1000.0, 1.0, 100.0, 1.2)])
        assert isclose(feq, 1200.0, rel_tol=1e-9)

    def test_invalid_duty_cycle_raises(self):
        """Empty cycles and bad segments are rejected."""
        bearing = make_ball_bearing()
        with pytest.raises(ValueError):
            bearing.equivalent_steady_load([])
        with pytest.raises(ValueError):
            bearing.equivalent_steady_load([(1000.0, -0.5)])
        with pytest.raises(ValueError):
            bearing.equivalent_steady_load([(1000.0, 0.5, 300.0, 1.0, 9.0)])
