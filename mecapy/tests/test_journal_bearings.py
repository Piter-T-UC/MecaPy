"""Tests for journal bearing and lubrication module."""

import math
from math import isclose

import pytest
from mecapy.bearings import JournalBearing
from mecapy.bearings.lubrication_data import (
    is_thick_film,
    raimondi_boyd,
    viscosity,
)


def make_journal(**kwargs):
    """Synthetic bearing with clean numbers: P = 1 MPa, S = 0.5."""
    defaults = dict(
        radius=25.0,
        clearance=0.025,
        length=50.0,
        speed=25.0,
        load=2500.0,
        viscosity=20.0,
    )
    defaults.update(kwargs)
    return JournalBearing(**defaults)


class TestViscosity:
    """SAE viscosity-temperature fit (fig. 12-13)."""

    def test_sae30_anchor(self):
        """SAE 30 at 70 degC is about 21 mPa*s."""
        assert isclose(viscosity(30, 70.0), 21.0, rel_tol=2e-2)

    def test_monotonic_decrease(self):
        """Viscosity falls with temperature for every grade."""
        for grade in (10, 20, 30, 40, 50, 60):
            values = [viscosity(grade, t) for t in (20.0, 60.0, 100.0, 140.0)]
            assert all(a > b for a, b in zip(values, values[1:]))

    def test_heavier_grade_is_thicker(self):
        """At one temperature, higher SAE grades are more viscous."""
        assert viscosity(60, 80.0) > viscosity(30, 80.0) > viscosity(10, 80.0)

    def test_unknown_grade_raises(self):
        """Unknown grades are rejected listing the available ones."""
        with pytest.raises(ValueError):
            viscosity(25, 70.0)


class TestPetroffSommerfeld:
    """Petroff friction (eq. 12-6) and Sommerfeld number (eq. 12-7)."""

    def test_pressure(self):
        """P = W / (2 r l) in MPa."""
        assert isclose(make_journal().pressure, 1.0, rel_tol=1e-9)

    def test_sommerfeld(self):
        """S = (r/c)^2 * mu*N/P with the synthetic numbers gives 0.5."""
        assert isclose(make_journal().sommerfeld, 0.5, rel_tol=1e-9)

    def test_sommerfeld_scales_with_viscosity(self):
        """Doubling the viscosity doubles S."""
        assert isclose(make_journal(viscosity=40.0).sommerfeld, 1.0, rel_tol=1e-9)

    def test_petroff_friction(self):
        """f = 2 pi^2 (mu N / P)(r/c) = 9.87e-3 for the synthetic case."""
        f = make_journal().petroff_friction()
        assert isclose(f, 2.0 * math.pi**2 * (0.02 * 25.0 / 1e6) * 1000.0, rel_tol=1e-9)
        assert isclose(f, 9.8696e-3, rel_tol=1e-4)

    def test_petroff_torque(self):
        """T = f W r in N*mm."""
        journal = make_journal()
        assert isclose(
            journal.petroff_friction_torque(),
            journal.petroff_friction() * 2500.0 * 25.0,
            rel_tol=1e-9,
        )

    def test_thick_film_check(self):
        """Fig. 12-4 boundary separates thick from mixed-film operation."""
        # Bearing modulus 20 cP * 1500 rpm / 145 psi ~ 207 >> 30
        assert make_journal().is_thick_film()
        # Hot, thin oil under heavy pressure: 1 * 300 / 2900 ~ 0.1 < 30
        assert not is_thick_film(1.0, 5.0, 20.0)


class TestRaimondiBoyd:
    """Raimondi-Boyd chart interpolation (figs. 12-16 to 12-24)."""

    def test_exact_table_point(self):
        """S = 0.121 at l/d = 1 reproduces the digitized row."""
        result = raimondi_boyd(0.121, 1.0)
        assert isclose(result["h0_over_c"], 0.4, rel_tol=1e-3)
        assert isclose(result["phi_deg"], 50.58, rel_tol=1e-3)
        assert isclose(result["friction_variable"], 3.22, rel_tol=1e-3)
        assert isclose(result["flow_variable"], 4.33, rel_tol=1e-3)
        assert isclose(result["side_flow_ratio"], 0.680, rel_tol=1e-3)
        assert isclose(result["p_over_pmax"], 0.415, rel_tol=1e-3)

    def test_between_points_is_bracketed(self):
        """Interpolated values lie between the neighbouring rows."""
        result = raimondi_boyd(0.2, 1.0)
        assert 0.4 < result["h0_over_c"] < 0.6
        assert 3.22 < result["friction_variable"] < 5.79
        assert 0.497 < result["side_flow_ratio"] < 0.680

    def test_infinite_bearing(self):
        """The long-bearing table has zero side flow."""
        result = raimondi_boyd(0.0626, math.inf)
        assert isclose(result["h0_over_c"], 0.6, rel_tol=1e-3)
        assert result["side_flow_ratio"] == 0.0

    def test_eq_12_16_collapses_to_tables(self):
        """The l/d blending formula reduces to the tables at 1/4, 1/2, 1."""
        direct = raimondi_boyd(0.319, 0.5)
        # Force the blending path with an l/d infinitesimally off 0.5
        blended = raimondi_boyd(0.319, 0.5 + 1e-7)
        for field in direct:
            assert isclose(direct[field], blended[field], rel_tol=1e-4)

    def test_intermediate_l_over_d_bracketed(self):
        """l/d = 0.75 falls between the 0.5 and 1.0 solutions."""
        low = raimondi_boyd(0.2, 0.5)
        mid = raimondi_boyd(0.2, 0.75)
        high = raimondi_boyd(0.2, 1.0)
        for field in ("h0_over_c", "friction_variable", "flow_variable"):
            lo, hi = sorted((low[field], high[field]))
            assert lo <= mid[field] <= hi

    def test_invalid_inputs_raise(self):
        """Non-positive S and sub-chart l/d are rejected."""
        with pytest.raises(ValueError):
            raimondi_boyd(0.0, 1.0)
        with pytest.raises(ValueError):
            raimondi_boyd(0.1, 0.1)


class TestJournalBearing:
    """JournalBearing construction and performance."""

    def test_constructor_validation(self):
        """Bad geometry, speed and lubricant specs are rejected."""
        with pytest.raises(ValueError):
            make_journal(clearance=30.0)  # clearance >= radius
        with pytest.raises(ValueError):
            make_journal(speed=0.0)
        with pytest.raises(ValueError):
            make_journal(load=-1.0)
        with pytest.raises(ValueError):
            JournalBearing(25.0, 0.025, 50.0, 25.0, 2500.0)  # no lubricant
        with pytest.raises(ValueError):
            JournalBearing(
                25.0,
                0.025,
                50.0,
                25.0,
                2500.0,
                viscosity=20.0,
                sae_grade=30,
                temperature=70.0,
            )

    def test_sae_grade_lubricant(self):
        """sae_grade + temperature resolves through the fig. 12-13 fit."""
        journal = make_journal(viscosity=None, sae_grade=30, temperature=70.0)
        assert isclose(journal.viscosity, viscosity(30, 70.0), rel_tol=1e-9)

    def test_setters_revalidate_and_keep_invariant(self):
        """Mutating an input after construction re-runs its validation."""
        journal = make_journal()
        journal.load = 5000.0  # valid change accepted
        assert journal.load == 5000.0
        with pytest.raises(ValueError):
            journal.clearance = journal.radius  # clearance must stay < radius
        with pytest.raises(ValueError):
            journal.speed = 0
        with pytest.raises(ValueError):
            journal.viscosity = -1

    def test_l_over_d(self):
        """l/d = length / (2 * radius)."""
        assert isclose(make_journal().l_over_d, 1.0, rel_tol=1e-9)

    def test_performance_internal_consistency(self):
        """Physical outputs follow from the chart variables and geometry."""
        journal = make_journal()
        perf = journal.performance()
        assert isclose(perf["h0"], perf["h0_over_c"] * 0.025, rel_tol=1e-9)
        assert isclose(
            perf["eccentricity_ratio"], 1.0 - perf["h0_over_c"], rel_tol=1e-9
        )
        assert isclose(
            perf["friction_coefficient"],
            perf["friction_variable"] * 0.025 / 25.0,
            rel_tol=1e-9,
        )
        assert isclose(
            perf["friction_torque"],
            perf["friction_coefficient"] * 2500.0 * 25.0,
            rel_tol=1e-9,
        )
        assert isclose(perf["pmax"], 1.0 / perf["p_over_pmax"], rel_tol=1e-9)
        assert isclose(
            perf["side_flow"], perf["side_flow_ratio"] * perf["flow"], rel_tol=1e-9
        )
        # S = 0.5 at l/d = 1 lies between the 0.264 and 0.631 rows
        assert 0.6 < perf["h0_over_c"] < 0.8

    def test_power_loss_positive(self):
        """Friction power loss is positive and consistent with torque."""
        journal = make_journal()
        perf = journal.performance()
        assert isclose(
            perf["power_loss"],
            2.0 * math.pi * 25.0 * perf["friction_torque"] / 1000.0,
            rel_tol=1e-9,
        )
        assert perf["power_loss"] > 0

    def test_temperature_rise_scales_with_pressure(self):
        """dT is positive and linear in P at fixed chart variables."""
        journal = make_journal()
        dt = journal.temperature_rise()
        assert dt > 0
        chart = raimondi_boyd(journal.sommerfeld, journal.l_over_d)
        expected = (
            8.30
            * 1.0
            * chart["friction_variable"]
            / (chart["flow_variable"] * (1.0 - 0.5 * chart["side_flow_ratio"]))
        )
        assert isclose(dt, expected, rel_tol=1e-9)

    def test_trumpler_check(self):
        """Criteria dict flags pass/fail and skips absent inputs."""
        journal = make_journal()
        result = journal.trumpler_check(startup_load=2500.0, max_temperature=70.0)
        # h0 ~ 0.65 * 0.025 = 0.016 mm vs limit 0.005 + 0.00004*50 = 0.007 mm
        assert result["min_film"] is True
        assert result["startup_pressure"] is True  # 1 MPa <= 2.07 MPa
        assert result["max_temperature"] is True
        assert result["design_factor_film"] in (True, False)
        skipped = journal.trumpler_check()
        assert skipped["startup_pressure"] is None
        assert skipped["max_temperature"] is None

    def test_repr(self):
        """Repr names the class and key geometry."""
        assert "JournalBearing" in repr(make_journal())
