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

    def test_setters_revalidate_and_keep_invariant(self):
        """Mutating a dimension/rating re-runs the constructor's checks."""
        bearing = make_ball_bearing()
        bearing.bore_diameter = 30.0  # valid: below the outer diameter
        assert bearing.bore_diameter == 30.0
        with pytest.raises(ValueError):
            bearing.outer_diameter = 10.0  # not larger than the bore
        with pytest.raises(ValueError):
            bearing.width = 0
        with pytest.raises(ValueError):
            bearing.C10 = -1

    def test_life_exponent_recomputes_on_type_change(self):
        """Changing bearing_type updates life_exponent (and life) at once."""
        bearing = make_ball_bearing()
        assert bearing.life_exponent == 3.0
        bearing.bearing_type = "roller"
        assert bearing.life_exponent == pytest.approx(10.0 / 3.0)
        assert isclose(bearing.life(7000.0), 1e6 * 5.0 ** (10.0 / 3.0), rel_tol=1e-4)
        with pytest.raises(ValueError):
            bearing.bearing_type = "magnetic"

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


class TestXYTableGate:
    """Table 11-1 applies to the ball family, keyed on type not exponent."""

    def test_ball_family_accepted(self):
        """Both a = 3 types go through the X/Y table."""
        for bearing_type in ("ball", "angular_contact"):
            bearing = make_ball_bearing(bearing_type=bearing_type)
            assert bearing.equivalent_load(3000.0, 1400.0) > 0

    def test_roller_family_rejected(self):
        """Roller families under thrust still need manufacturer data."""
        for bearing_type in ("roller", "cylindrical", "tapered"):
            bearing = make_ball_bearing(bearing_type=bearing_type)
            with pytest.raises(ValueError):
                bearing.equivalent_load(3000.0, 1400.0)
            # ...but pure radial load is fine
            assert bearing.equivalent_load(3000.0) == 3000.0


class TestBearingPintInputs:
    """Optional pint quantities at the boundary (plain floats unchanged)."""

    def test_dimensions_and_ratings_accept_quantities(self):
        """An inch bore and a kN rating land on mm and N."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        bearing = make_ball_bearing(
            bore_diameter=1 * ureg.inch, C10=35 * ureg.kN, C0=14 * ureg.kN
        )
        assert bearing.bore_diameter == pytest.approx(25.4)
        assert bearing.C10 == pytest.approx(35000.0)
        assert bearing.C0 == pytest.approx(14000.0)

    def test_loads_and_speeds_accept_quantities(self):
        """Life is identical whether fed Quantities or plain floats."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        bearing = make_ball_bearing()
        assert bearing.life(7 * ureg.kN) == pytest.approx(
            bearing.life(7000.0), rel=1e-12
        )
        assert bearing.life_hours(7000.0, 1000 * ureg.rpm) == pytest.approx(
            bearing.life_hours(7000.0, 1000.0), rel=1e-12
        )
        assert bearing.equivalent_load(3 * ureg.kN, 1400 * ureg.N) == pytest.approx(
            bearing.equivalent_load(3000.0, 1400.0), rel=1e-12
        )

    def test_duty_cycle_segments_accept_quantities(self):
        """Per-segment loads and speeds convert element-wise."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        bearing = make_ball_bearing()
        mixed = [(3 * ureg.kN, 0.5, 1000 * ureg.rpm), (5000.0, 0.5, 1500.0)]
        plain = [(3000.0, 0.5, 1000.0), (5000.0, 0.5, 1500.0)]
        assert bearing.equivalent_steady_load(mixed) == pytest.approx(
            bearing.equivalent_steady_load(plain), rel=1e-12
        )

    def test_wrong_dimension_is_rejected(self):
        """A force where a length belongs raises pint's DimensionalityError."""
        pint = pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        with pytest.raises(pint.DimensionalityError):
            make_ball_bearing(bore_diameter=5 * ureg.newton)

    def test_plain_floats_still_validate(self):
        """The pint boundary does not weaken the positivity checks."""
        with pytest.raises(ValueError):
            make_ball_bearing(C10=-1.0)
        bearing = make_ball_bearing()
        with pytest.raises(ValueError):
            bearing.rating_life = 0.0


class TestMeanDiameterAndDescribe:
    """Geometry helper and the house report."""

    def test_mean_diameter(self):
        assert make_ball_bearing().mean_diameter == pytest.approx(38.5)

    def test_describe_returns_string_with_labels(self):
        """describe() returns (never prints) a labelled multi-line report."""
        text = make_ball_bearing(name="6205").describe()
        assert isinstance(text, str)
        assert "Bearing geometry '6205'" in text
        assert "bore diameter (d) = 25.000 mm" in text
        assert "mean diameter (dm) = 38.500 mm" in text
        assert "dynamic rating (C10) = 35000.0 N" in text

    def test_describe_without_ratings(self):
        """Missing ratings degrade gracefully instead of raising."""
        bearing = Bearing(25.0, 52.0, 15.0)
        text = bearing.describe()
        assert "dynamic rating (C10) = not given" in text
        assert "static rating (C0) = not given" in text


class TestStaticRating:
    """ISO 76 static equivalent load and static safety factor."""

    def test_p0_floor_binds(self):
        """0.6*3000 + 0.5*1400 = 2500 N, floored at Fr = 3000 N."""
        bearing = make_ball_bearing()
        assert bearing.static_equivalent_load(3000.0, 1400.0) == pytest.approx(3000.0)
        assert bearing.static_safety_factor(3000.0, 1400.0) == pytest.approx(
            14000.0 / 3000.0
        )

    def test_p0_floor_does_not_bind(self):
        """0.6*3000 + 0.5*4000 = 3800 N > Fr, so the sum governs."""
        bearing = make_ball_bearing()
        assert bearing.static_equivalent_load(3000.0, 4000.0) == pytest.approx(3800.0)
        assert bearing.static_safety_factor(3000.0, 4000.0) == pytest.approx(
            14000.0 / 3800.0
        )

    def test_tapered_needs_contact_angle(self):
        """Y0 = 0.22*cot(alpha) cannot be evaluated without alpha."""
        bearing = make_ball_bearing(bearing_type="tapered")
        with pytest.raises(ValueError):
            bearing.static_equivalent_load(3000.0, 1000.0)
        expected = max(0.5 * 3000.0 + 0.8211 * 1000.0, 3000.0)
        assert bearing.static_equivalent_load(
            3000.0, 1000.0, contact_angle_deg=15.0
        ) == pytest.approx(expected, rel=1e-3)

    def test_invalid_inputs(self):
        bearing = make_ball_bearing()
        with pytest.raises(ValueError):
            bearing.static_equivalent_load(-1.0)
        with pytest.raises(ValueError):
            Bearing(25.0, 52.0, 15.0).static_safety_factor(1000.0)


class TestISO281:
    """Modified rating life L_nm = a1 * a_ISO * L10 (ISO 281:2007)."""

    def test_reference_viscosity_closed_form(self):
        """nu1 matches the ISO closed form at dm = 38.5 mm, n = 1500 rpm."""
        from mecapy.bearings.iso281_data import reference_viscosity

        assert reference_viscosity(38.5, 1500.0) == pytest.approx(
            4500.0 * 1500.0**-0.5 * 38.5**-0.5, rel=1e-12
        )
        # the low-speed branch uses a different fit
        assert reference_viscosity(38.5, 500.0) == pytest.approx(
            45000.0 * 500.0**-0.83 * 38.5**-0.5, rel=1e-12
        )

    def test_kappa_falls_with_temperature(self):
        """Hotter oil is thinner, so the film quality drops."""
        bearing = make_ball_bearing()
        cold = bearing.viscosity_ratio(1500.0, sae_grade=30, temperature=50.0)
        hot = bearing.viscosity_ratio(1500.0, sae_grade=30, temperature=100.0)
        assert cold > hot

    def test_lubricant_must_be_specified_exactly_once(self):
        bearing = make_ball_bearing()
        with pytest.raises(ValueError):
            bearing.viscosity_ratio(1500.0)
        with pytest.raises(ValueError):
            bearing.viscosity_ratio(1500.0, kinematic_viscosity=30.0, sae_grade=30)

    def test_a_iso_monotone_in_kappa_and_cleanliness(self):
        """More viscosity and more cleanliness both buy life."""
        from mecapy.bearings.iso281_data import a_iso

        assert a_iso(0.1, 0.5) < a_iso(0.1, 2.0)
        bearing = make_ball_bearing()
        dirty = bearing.life_modification_factor(
            8000.0,
            1500.0,
            contamination="severe_contamination",
            sae_grade=30,
            temperature=70.0,
        )
        clean = bearing.life_modification_factor(
            8000.0,
            1500.0,
            contamination="high_cleanliness",
            sae_grade=30,
            temperature=70.0,
        )
        assert dirty < clean

    def test_a_iso_is_clamped(self):
        """The standard caps a_ISO at 50 and the model floors it at 0.1."""
        from mecapy.bearings.iso281_data import A_ISO_MAX, A_ISO_MIN, a_iso

        assert a_iso(10.0, 4.0) == pytest.approx(A_ISO_MAX)
        assert a_iso(0.0, 1.0) == pytest.approx(A_ISO_MIN)

    def test_iso_life_is_a1_times_a_iso_times_l10(self):
        """The composition is exactly the ISO 281 product."""
        from mecapy.bearings.iso281_data import get_reliability_factor

        bearing = make_ball_bearing()
        kwargs = dict(sae_grade=30, temperature=70.0)
        a_iso_factor = bearing.life_modification_factor(3000.0, 1500.0, **kwargs)
        expected = get_reliability_factor(0.99) * a_iso_factor * bearing.life(3000.0)
        assert bearing.iso_life(
            3000.0, 1500.0, reliability=0.99, **kwargs
        ) == pytest.approx(expected, rel=1e-12)

    def test_iso_life_hours_conversion(self):
        bearing = make_ball_bearing()
        kwargs = dict(sae_grade=30, temperature=70.0)
        assert bearing.iso_life_hours(3000.0, 1500.0, **kwargs) == pytest.approx(
            bearing.iso_life(3000.0, 1500.0, **kwargs) / (60.0 * 1500.0), rel=1e-12
        )

    def test_a1_at_r90_is_one(self):
        from mecapy.bearings.iso281_data import get_reliability_factor

        assert get_reliability_factor(0.90) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            get_reliability_factor(0.5)

    def test_fatigue_load_limit_explicit_beats_estimate(self):
        """An explicit catalog Cu always wins over the C0/8.2 rule."""
        bearing = make_ball_bearing()
        assert bearing.fatigue_load_limit() == pytest.approx(14000.0 / 8.2)
        assert bearing.fatigue_load_limit(Cu=900.0) == pytest.approx(900.0)
        with pytest.raises(ValueError):
            Bearing(25.0, 52.0, 15.0).fatigue_load_limit()


class TestSpeedLimits:
    """n*dm feasibility check."""

    def test_dn_and_limiting_speed_round_trip(self):
        bearing = make_ball_bearing()
        result = bearing.speed_check(bearing.speed_limit())
        assert result["margin"] == pytest.approx(1.0, rel=1e-12)
        assert result["within_limit"] is True

    def test_overspeed_is_flagged(self):
        bearing = make_ball_bearing()
        overspeed = bearing.speed_check(10.0 * bearing.speed_limit())
        assert overspeed["within_limit"] is False

    def test_oil_allows_more_than_grease(self):
        bearing = make_ball_bearing()
        assert bearing.speed_limit("oil") > bearing.speed_limit("grease")
        with pytest.raises(ValueError):
            bearing.speed_limit("magic")


class TestTaperedThrust:
    """Induced thrust and the opposed-pair rule (Shigley sec. 11-11)."""

    def test_induced_thrust_hand_check(self):
        """Fi = 0.47 * 3000 / 1.5 = 940 N."""
        assert make_ball_bearing().induced_thrust(3000.0) == pytest.approx(940.0)

    def test_pair_case_switch_is_continuous(self):
        """Both branches agree at the switching thrust."""
        fr_a, fr_b = 4000.0, 2000.0
        switch = 0.47 * fr_a / 1.5 - 0.47 * fr_b / 1.5
        low = Bearing.tapered_pair_loads(fr_a, fr_b, switch - 1e-9)
        high = Bearing.tapered_pair_loads(fr_a, fr_b, switch + 1e-9)
        assert low[0] == pytest.approx(high[0], rel=1e-6)
        assert low[1] == pytest.approx(high[1], rel=1e-6)

    def test_pair_loads_floor_at_radial(self):
        """Neither row is ever rated below its own radial load."""
        fe_a, fe_b = Bearing.tapered_pair_loads(3000.0, 2000.0, 500.0)
        assert fe_a >= 3000.0
        assert fe_b >= 2000.0

    def test_invalid_inputs(self):
        bearing = make_ball_bearing()
        with pytest.raises(ValueError):
            bearing.induced_thrust(-1.0)
        with pytest.raises(ValueError):
            bearing.induced_thrust(1000.0, K=0.0)


class TestDutyCycleReport:
    """Per-segment damage breakdown."""

    def test_damage_shares_sum_to_one(self):
        bearing = make_ball_bearing()
        report = bearing.duty_cycle_report(
            [(3000.0, 0.5, 1000.0), (5000.0, 0.5, 1500.0)]
        )
        assert report["damage_total"] == pytest.approx(1.0, rel=1e-12)
        assert len(report["segments"]) == 2

    def test_equivalent_load_matches_cubic_mean(self):
        bearing = make_ball_bearing()
        duty = [(3000.0, 0.5, 1000.0), (5000.0, 0.5, 1500.0)]
        report = bearing.duty_cycle_report(duty)
        assert report["equivalent_load"] == pytest.approx(
            bearing.equivalent_steady_load(duty), rel=1e-12
        )
        assert report["life"] == pytest.approx(
            bearing.life(report["equivalent_load"]), rel=1e-12
        )

    def test_severe_segment_dominates_damage(self):
        """The heavier segment carries most of the damage at equal time."""
        bearing = make_ball_bearing()
        report = bearing.duty_cycle_report(
            [(2000.0, 0.5, 1000.0), (6000.0, 0.5, 1000.0)]
        )
        assert report["segments"][1]["damage_share"] > 0.9

    def test_report_without_c10(self):
        """Lives degrade to None rather than raising."""
        bearing = Bearing(25.0, 52.0, 15.0)
        report = bearing.duty_cycle_report([(3000.0, 1.0)])
        assert report["life"] is None
        assert report["segments"][0]["life"] is None


class TestCoverageBackfill:
    """Paths that had no test before the subsystem was extended."""

    def test_rating_life_setter_rejects_nonpositive(self):
        bearing = make_ball_bearing()
        with pytest.raises(ValueError):
            bearing.rating_life = -1.0

    def test_life_hours_rejects_nonpositive_speed(self):
        with pytest.raises(ValueError):
            make_ball_bearing().life_hours(7000.0, 0.0)

    def test_reliability_rejects_negative_life(self):
        with pytest.raises(ValueError):
            make_ball_bearing().reliability(7000.0, -1.0)

    def test_weibull_reliability_edges(self):
        """Below x0 = 0.02 nothing has failed yet; negative x is rejected."""
        assert weibull_reliability(0.0) == pytest.approx(1.0)
        assert weibull_reliability(0.01) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            weibull_reliability(-1.0)

    def test_bearing_uses_mecha_element_material_behaviour(self):
        """The inherited SI stress helpers work on a Bearing too."""
        bearing = make_ball_bearing()
        assert bearing.material_properties["yield_strength"] > 0
        stress = bearing.calculate_stress(1000.0, 50.0)  # N over mm^2 -> MPa
        assert stress == pytest.approx(20.0)
        with pytest.raises(ValueError):
            bearing.calculate_stress(1000.0, 0.0)

    def test_every_export_is_importable(self):
        """__all__ and the module contents agree."""
        import mecapy.bearings as bearings

        for name in bearings.__all__:
            assert hasattr(bearings, name), name

    def test_fixtures_are_wired_up(self, sample_ball_bearing, sample_bushing):
        assert sample_ball_bearing.life(7000.0) == pytest.approx(125e6)
        assert sample_bushing.pv == pytest.approx(0.62832, rel=1e-4)


class TestISO281DataModule:
    """Accessors and validation of the ISO 281/76 data module."""

    def test_reference_viscosity_validation(self):
        from mecapy.bearings.iso281_data import reference_viscosity

        with pytest.raises(ValueError):
            reference_viscosity(0.0, 1500.0)
        with pytest.raises(ValueError):
            reference_viscosity(38.5, 0.0)

    def test_viscosity_ratio_validation(self):
        from mecapy.bearings.iso281_data import viscosity_ratio

        with pytest.raises(ValueError):
            viscosity_ratio(0.0, 38.5, 1500.0)

    def test_kinematic_from_dynamic(self):
        """nu = mu/rho, with mPa*s and kg/m^3 in and mm^2/s out."""
        from mecapy.bearings.iso281_data import kinematic_from_dynamic

        assert kinematic_from_dynamic(87.0, density=870.0) == pytest.approx(100.0)
        with pytest.raises(ValueError):
            kinematic_from_dynamic(0.0)
        with pytest.raises(ValueError):
            kinematic_from_dynamic(30.0, density=0.0)

    def test_contamination_factor_bands(self):
        """The table splits at dm = 100 mm and offers min/mid/max."""
        from mecapy.bearings.iso281_data import get_contamination_factor

        small = get_contamination_factor("normal_cleanliness", 50.0, "min")
        large = get_contamination_factor("normal_cleanliness", 150.0, "min")
        assert large > small  # bigger bearings tolerate more contamination
        assert get_contamination_factor("normal_cleanliness", 50.0, "max") == (
            pytest.approx(0.6)
        )
        assert get_contamination_factor("extreme_cleanliness", 50.0) == (
            pytest.approx(1.0)
        )
        with pytest.raises(ValueError):
            get_contamination_factor("spotless", 50.0)
        with pytest.raises(ValueError):
            get_contamination_factor("normal_cleanliness", 0.0)
        with pytest.raises(ValueError):
            get_contamination_factor("normal_cleanliness", 50.0, "middling")

    def test_static_factor_lookup(self):
        from mecapy.bearings.iso281_data import get_static_factors

        assert get_static_factors("ball") == (0.6, 0.5)
        assert get_static_factors("cylindrical") == (1.0, 0.0)
        with pytest.raises(ValueError):
            get_static_factors("magnetic")
        with pytest.raises(ValueError):
            get_static_factors("angular_contact")  # needs a contact angle
        with pytest.raises(ValueError):
            get_static_factors("angular_contact", contact_angle_deg=0.0)
        with pytest.raises(ValueError):
            get_static_factors("angular_contact", contact_angle_deg=90.0)

    def test_limiting_dn_lookup(self):
        from mecapy.bearings.iso281_data import get_limiting_dn

        assert get_limiting_dn("ball", "oil") > get_limiting_dn("ball", "grease")
        with pytest.raises(ValueError):
            get_limiting_dn("magnetic")
        with pytest.raises(ValueError):
            get_limiting_dn("ball", "water")

    def test_a_iso_validation_and_families(self):
        from mecapy.bearings.iso281_data import a_iso, get_a_iso_family

        with pytest.raises(ValueError):
            a_iso(-1.0, 1.0)
        with pytest.raises(ValueError):
            a_iso(0.5, 1.0, family="radial_magnetic")
        assert get_a_iso_family("ball") == "radial_ball"
        assert get_a_iso_family("tapered") == "radial_roller"
        with pytest.raises(ValueError):
            get_a_iso_family("magnetic")

    def test_a_iso_kappa_is_clamped_to_the_valid_band(self):
        """Outside 0.1 to 4 the curves are undefined, so the ends are used."""
        from mecapy.bearings.iso281_data import a_iso

        assert a_iso(0.2, 0.01) == pytest.approx(a_iso(0.2, 0.1))
        assert a_iso(0.2, 100.0) == pytest.approx(a_iso(0.2, 4.0))

    def test_roller_family_uses_its_own_coefficients(self):
        """Ball and roller families give different factors at one point."""
        from mecapy.bearings.iso281_data import a_iso

        assert a_iso(0.1, 1.0, family="radial_ball") != pytest.approx(
            a_iso(0.1, 1.0, family="radial_roller")
        )
