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


class TestChartRefactor:
    """One chart evaluation per check, with identical numbers (D: perf split)."""

    def test_design_factor_branch_matches_explicit_overload(self):
        """S ~ 1/W scaling reproduces a bearing rebuilt at n_d * W exactly."""
        journal = make_journal()
        design_factor = 2.0
        overloaded = make_journal(load=design_factor * journal.load)
        limit = 0.005 + 0.00004 * 2.0 * journal.radius
        expected = overloaded.performance()["h0"] >= limit
        assert (
            journal.trumpler_check(design_factor=design_factor)["design_factor_film"]
            is expected
        )
        # and the underlying film thickness itself, not just the verdict
        chart = raimondi_boyd(journal.sommerfeld / design_factor, journal.l_over_d)
        assert chart["h0_over_c"] * journal.clearance == pytest.approx(
            overloaded.performance()["h0"], rel=1e-12
        )

    def test_performance_from_chart_matches_performance(self):
        """The private chart path equals the public one (rel=1e-12)."""
        journal = make_journal()
        chart = raimondi_boyd(journal.sommerfeld, journal.l_over_d)
        direct = journal._performance_from_chart(chart)
        for key, value in journal.performance().items():
            assert direct[key] == pytest.approx(value, rel=1e-12)

    def test_temperature_rise_from_chart_matches(self):
        """Same for the temperature-rise helper."""
        journal = make_journal()
        chart = raimondi_boyd(journal.sommerfeld, journal.l_over_d)
        assert journal._temperature_rise_from_chart(chart) == pytest.approx(
            journal.temperature_rise(), rel=1e-12
        )

    def test_trumpler_invalid_arguments(self):
        """Non-positive design factor and startup load are rejected."""
        journal = make_journal()
        with pytest.raises(ValueError):
            journal.trumpler_check(design_factor=0.0)
        with pytest.raises(ValueError):
            journal.trumpler_check(startup_load=-1.0)


class TestLubricantProvenance:
    """The bearing remembers how its lubricant was specified."""

    def test_sae_path_stores_grade_and_temperature(self):
        journal = make_journal(viscosity=None, sae_grade=40, temperature=60.0)
        assert journal.sae_grade == 40
        assert journal.film_temperature == pytest.approx(60.0)
        assert journal.viscosity == pytest.approx(viscosity(40, 60.0))

    def test_explicit_viscosity_leaves_grade_none(self):
        journal = make_journal()
        assert journal.sae_grade is None
        assert journal.film_temperature is None


class TestJournalPintInputs:
    """Optional pint quantities at the boundary (plain floats unchanged)."""

    def test_dimensions_accept_quantities(self):
        """An inch/rpm/lbf bearing lands on the same state as mm/rev-s/N."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        quantity = JournalBearing(
            radius=1 * ureg.inch,
            clearance=0.025 * ureg.mm,
            length=2 * ureg.inch,
            speed=1500 * ureg.rpm,
            load=2500 * ureg.N,
            viscosity=20 * ureg.cP,
        )
        plain = JournalBearing(25.4, 0.025, 50.8, 25.0, 2500.0, viscosity=20.0)
        assert quantity.radius == pytest.approx(plain.radius)
        assert quantity.speed == pytest.approx(plain.speed)
        assert quantity.viscosity == pytest.approx(plain.viscosity)
        assert quantity.sommerfeld == pytest.approx(plain.sommerfeld, rel=1e-12)

    def test_wrong_dimension_is_rejected(self):
        """A force where a length belongs raises pint's DimensionalityError."""
        pint = pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        with pytest.raises(pint.DimensionalityError):
            make_journal(radius=5 * ureg.newton)

    def test_plain_floats_still_validate(self):
        """The pint boundary does not weaken the positivity checks."""
        with pytest.raises(ValueError):
            make_journal(load=-1.0)
        journal = make_journal()
        with pytest.raises(ValueError):
            journal.viscosity = 0.0


class TestSommerfeldInverse:
    """Inverse of the h0/c chart column."""

    def test_round_trip_on_tabulated_ratios(self):
        """S recovers the tabulated row exactly at l/d = 1."""
        from mecapy.bearings.lubrication_data import sommerfeld_for

        assert sommerfeld_for(0.4, 1.0) == pytest.approx(0.121, rel=1e-6)
        assert sommerfeld_for(0.4, 0.5) == pytest.approx(0.319, rel=1e-6)

    def test_round_trip_on_blended_ratio(self):
        """A blended l/d inverts to the ratio it came from."""
        from mecapy.bearings.lubrication_data import raimondi_boyd, sommerfeld_for

        sommerfeld = sommerfeld_for(0.35, 0.75)
        assert raimondi_boyd(sommerfeld, 0.75)["h0_over_c"] == pytest.approx(
            0.35, rel=1e-9
        )

    def test_out_of_range_raises(self):
        from mecapy.bearings.lubrication_data import sommerfeld_for

        with pytest.raises(ValueError):
            sommerfeld_for(0.0, 1.0)
        with pytest.raises(ValueError):
            sommerfeld_for(1.0, 1.0)
        with pytest.raises(ValueError):
            sommerfeld_for(0.98, 1.0)  # above the charted maximum


class TestThermalSolve:
    """Self-consistent mean film temperature (Shigley sec. 12-8)."""

    def make_sae_journal(self):
        """The example-script bearing: SAE 40, 60 degC inlet."""
        return JournalBearing(
            25.0, 0.025, 50.0, 25.0, 2500.0, sae_grade=40, temperature=60.0
        )

    def test_matches_the_damped_hand_loop(self):
        """Reproduces the loop that used to live in examples/."""
        journal = self.make_sae_journal()
        expected = 60.0
        for _ in range(60):
            trial = JournalBearing(
                25.0, 0.025, 50.0, 25.0, 2500.0, sae_grade=40, temperature=expected
            )
            expected = 0.5 * (expected + (60.0 + trial.temperature_rise() / 2.0))
        result = journal.solve_film_temperature(60.0)
        assert result["temperature"] == pytest.approx(expected, abs=0.1)
        assert result["converged"] is True

    def test_fixed_point_residual(self):
        """At the answer, T_avg = T_in + dT/2."""
        journal = self.make_sae_journal()
        result = journal.solve_film_temperature(
            60.0, tolerance=1e-10, max_iterations=500
        )
        assert result["temperature"] == pytest.approx(
            60.0 + result["rise"] / 2.0, rel=1e-6
        )

    def test_relaxation_does_not_move_the_root(self):
        """Heavier and lighter damping land on the same fixed point."""
        slow = self.make_sae_journal().solve_film_temperature(
            60.0, relaxation=0.3, tolerance=1e-10, max_iterations=500
        )
        fast = self.make_sae_journal().solve_film_temperature(
            60.0, relaxation=0.9, tolerance=1e-10, max_iterations=500
        )
        assert slow["temperature"] == pytest.approx(fast["temperature"], rel=1e-4)

    def test_apply_writes_back_the_operating_point(self):
        """With apply=True the bearing becomes its own solved state."""
        journal = self.make_sae_journal()
        result = journal.solve_film_temperature(60.0)
        assert journal.viscosity == pytest.approx(result["viscosity"])
        assert journal.film_temperature == pytest.approx(result["temperature"])

    def test_apply_false_leaves_state(self):
        journal = self.make_sae_journal()
        before = journal.viscosity
        journal.solve_film_temperature(60.0, apply=False)
        assert journal.viscosity == pytest.approx(before)

    def test_hotter_inlet_gives_hotter_film(self):
        cool = self.make_sae_journal().solve_film_temperature(40.0)
        warm = self.make_sae_journal().solve_film_temperature(80.0)
        assert warm["temperature"] > cool["temperature"]

    def test_not_converged_flag(self):
        """A single iteration reports itself as unconverged."""
        result = self.make_sae_journal().solve_film_temperature(
            60.0, max_iterations=1, apply=False
        )
        assert result["converged"] is False
        assert result["iterations"] == 1

    def test_missing_grade_raises(self):
        """An explicit-viscosity bearing cannot re-evaluate mu(T)."""
        with pytest.raises(ValueError):
            make_journal().solve_film_temperature(60.0)
        # ...unless the grade is supplied at the call site
        assert make_journal().solve_film_temperature(60.0, sae_grade=40)["converged"]

    def test_invalid_parameters(self):
        journal = self.make_sae_journal()
        with pytest.raises(ValueError):
            journal.solve_film_temperature(60.0, relaxation=0.0)
        with pytest.raises(ValueError):
            journal.solve_film_temperature(60.0, relaxation=1.5)
        with pytest.raises(ValueError):
            journal.solve_film_temperature(60.0, tolerance=0.0)
        with pytest.raises(ValueError):
            journal.solve_film_temperature(60.0, max_iterations=0)


class TestDesignInverses:
    """Sizing one unknown for a target minimum film."""

    def test_viscosity_round_trip(self):
        """The returned viscosity delivers exactly the target film."""
        journal = make_journal()
        required = journal.viscosity_for_minimum_film(0.020)
        resized = make_journal(viscosity=required)
        assert resized.performance()["h0"] == pytest.approx(0.020, rel=1e-9)

    def test_length_round_trip(self):
        """The length iteration accounts for l/d moving with l."""
        journal = make_journal()
        required = journal.length_for_minimum_film(0.020)
        resized = make_journal(length=required)
        assert resized.performance()["h0"] == pytest.approx(0.020, rel=1e-6)

    def test_inverse_rejects_impossible_targets(self):
        journal = make_journal()
        with pytest.raises(ValueError):
            journal.viscosity_for_minimum_film(0.0)
        with pytest.raises(ValueError):
            journal.viscosity_for_minimum_film(journal.clearance)
        with pytest.raises(ValueError):
            journal.length_for_minimum_film(2.0 * journal.clearance)

    def test_minimum_film_safety_factor(self):
        journal = make_journal()
        assert journal.minimum_film_safety_factor() == pytest.approx(
            journal.performance()["h0"] / journal.minimum_film_limit, rel=1e-12
        )
        assert journal.minimum_film_limit == pytest.approx(0.005 + 0.00004 * 50.0)

    def test_film_for_clearance_matches_a_rebuilt_bearing(self):
        """The pure function agrees with actually changing the clearance."""
        journal = make_journal()
        assert journal.film_for_clearance(0.04) == pytest.approx(
            make_journal(clearance=0.04).performance()["h0"], rel=1e-12
        )
        assert journal.film_for_clearance(journal.clearance) == pytest.approx(
            journal.performance()["h0"], rel=1e-12
        )

    def test_optimum_clearance_is_an_interior_maximum(self):
        """h0 peaks at an intermediate clearance (the design trade-off)."""
        journal = make_journal()
        best = journal.optimum_clearance(0.005, 0.15)
        assert 0.005 < best < 0.15
        peak = journal.film_for_clearance(best)
        assert peak > journal.film_for_clearance(best * 0.5)
        assert peak > journal.film_for_clearance(best * 1.5)

    def test_clearance_window_brackets_the_optimum(self):
        journal = make_journal()
        best = journal.optimum_clearance(0.005, 0.15)
        low, high = journal.clearance_window_for_minimum_film(0.015, 0.005, 0.15)
        assert low < best < high

    def test_clearance_window_none_when_unreachable(self):
        journal = make_journal()
        assert journal.clearance_window_for_minimum_film(5.0, 0.005, 0.15) is None

    def test_clearance_sweep_validation(self):
        journal = make_journal()
        with pytest.raises(ValueError):
            journal.clearance_sweep(0.0, 0.1)
        with pytest.raises(ValueError):
            journal.clearance_sweep(0.1, 0.05)
        with pytest.raises(ValueError):
            journal.clearance_sweep(0.005, 30.0)  # beyond the radius
        with pytest.raises(ValueError):
            journal.clearance_sweep(0.005, 0.1, n=1)

    def test_clearance_sweep_shape(self):
        sweep = make_journal().clearance_sweep(0.01, 0.05, n=7)
        assert len(sweep["clearance"]) == 7
        assert len(sweep["h0"]) == 7
        assert sweep["clearance"][0] == pytest.approx(0.01)
        assert sweep["clearance"][-1] == pytest.approx(0.05)


class TestFilmGeometry:
    """h(theta) = c (1 + eps cos theta)."""

    def test_minimum_matches_performance(self):
        journal = make_journal()
        profile = journal.film_profile()
        assert min(profile["film_thickness"]) == pytest.approx(
            journal.performance()["h0"], rel=1e-12
        )
        assert profile["h0"] == pytest.approx(journal.performance()["h0"], rel=1e-12)

    def test_closed_form_at_a_sample(self):
        journal = make_journal()
        profile = journal.film_profile(n=5)  # 0, 90, 180, 270, 360 degrees
        eps = profile["eccentricity_ratio"]
        assert profile["film_thickness"][0] == pytest.approx(
            journal.clearance * (1.0 + eps)
        )
        assert profile["film_thickness"][2] == pytest.approx(
            journal.clearance * (1.0 - eps)
        )

    def test_eccentricity_property_matches_chart(self):
        journal = make_journal()
        assert journal.eccentricity_ratio == pytest.approx(
            journal.performance()["eccentricity_ratio"], rel=1e-12
        )

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            make_journal().film_profile(n=1)


class TestWhirlStability:
    """Half-frequency whirl rule of thumb."""

    def test_lightly_loaded_journal_is_whirl_prone(self):
        """A thick-film, low-eccentricity bearing is flagged."""
        journal = make_journal(viscosity=60.0)  # very stiff film, eps small
        assert journal.eccentricity_ratio < 0.6
        assert journal.is_whirl_prone is True
        assert journal.whirl_margin < 1.0

    def test_heavily_loaded_journal_is_stable(self):
        journal = make_journal(load=25000.0)  # eps driven up by the load
        assert journal.eccentricity_ratio > 0.6
        assert journal.is_whirl_prone is False
        assert journal.whirl_margin > 1.0

    def test_whirl_frequency_is_about_half_speed(self):
        journal = make_journal()
        assert journal.whirl_frequency == pytest.approx(0.47 * journal.speed)


class TestJournalDescribe:
    """The house report."""

    def test_describe_returns_labelled_string(self):
        text = make_journal(name="main").describe()
        assert isinstance(text, str)
        assert "JournalBearing geometry 'main'" in text
        assert "journal radius (r) = 25.000 mm" in text
        assert "Sommerfeld number (S) = 0.5000" in text
        assert "whirl margin" in text

    def test_describe_reports_the_lubricant_when_known(self):
        text = JournalBearing(
            25.0, 0.025, 50.0, 25.0, 2500.0, sae_grade=40, temperature=60.0
        ).describe()
        assert "lubricant = SAE 40 at 60.0 degC" in text
        assert "lubricant" not in make_journal().describe()


class TestJournalPlots:
    """Smoke tests: the plots return a matplotlib Figure."""

    @pytest.fixture(autouse=True)
    def _matplotlib(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        yield
        matplotlib.pyplot.close("all")

    def test_plot_film_returns_figure(self):
        from matplotlib.figure import Figure

        assert isinstance(make_journal().plot_film(show=False), Figure)

    def test_plot_clearance_design_returns_figure(self):
        from matplotlib.figure import Figure

        figure = make_journal().plot_clearance_design(show=False)
        assert isinstance(figure, Figure)

    def test_plot_accepts_existing_axes(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        assert make_journal().plot_film(show=False, ax=ax) is ax.figure


class TestCoverageBackfill:
    """Paths that had no test before the subsystem was extended."""

    def test_viscosity_reyn_matches_the_si_wrapper(self):
        """The imperial form is what the SI one is built on."""
        from mecapy.bearings.lubrication_data import (
            MPAS_PER_MICROREYN,
            viscosity_reyn,
        )

        assert MPAS_PER_MICROREYN * viscosity_reyn(30, 158.0) == pytest.approx(
            viscosity(30, 70.0), rel=1e-12
        )  # 158 degF = 70 degC
        with pytest.raises(ValueError):
            viscosity_reyn(25, 158.0)

    def test_raimondi_boyd_above_l_over_d_one(self):
        """l/d = 2 is bracketed by the l/d = 1 and infinite solutions."""
        long_bearing = raimondi_boyd(0.2, math.inf)
        wide = raimondi_boyd(0.2, 2.0)
        square = raimondi_boyd(0.2, 1.0)
        low, high = sorted((square["h0_over_c"], long_bearing["h0_over_c"]))
        assert low <= wide["h0_over_c"] <= high

    def test_fixtures_are_wired_up(self, sample_journal_bearing):
        """The shared conftest fixture builds the documented bearing."""
        assert sample_journal_bearing.pressure == pytest.approx(1.0)
        assert sample_journal_bearing.sommerfeld == pytest.approx(0.5)
