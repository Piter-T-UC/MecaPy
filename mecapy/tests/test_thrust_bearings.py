"""Tests for the hydrodynamic fixed-incline thrust bearing."""

import math

import pytest
from mecapy import MechaElement
from mecapy.bearings import ThrustBearing, load_coefficient
from mecapy.bearings.thrust import OPTIMUM_TAPER_RATIO, friction_coefficient_factor


def make_thrust(**kwargs):
    """Eight-pad collar: ri = 50, ro = 100 mm, 30 rev/s, 20 kN, SAE-ish oil."""
    defaults = dict(
        inner_radius=50.0,
        outer_radius=100.0,
        n_pads=8,
        speed=30.0,
        load=20000.0,
        viscosity=30.0,
    )
    defaults.update(kwargs)
    return ThrustBearing(**defaults)


class TestSliderCoefficients:
    """Closed-form plane-slider coefficients."""

    def test_load_coefficient_peaks_at_the_optimum(self):
        """Kw is maximized at a = 2.19, the classic tapered-land value."""
        peak = load_coefficient(OPTIMUM_TAPER_RATIO)
        assert peak > load_coefficient(1.5)
        assert peak > load_coefficient(3.5)
        assert OPTIMUM_TAPER_RATIO == pytest.approx(2.2, rel=1e-2)

    def test_coefficients_vanish_at_a_parallel_film(self):
        """A parallel film carries no load: Kw -> 0 as a -> 1."""
        assert load_coefficient(1.0001) < 1e-3

    def test_friction_factor_falls_with_taper(self):
        assert friction_coefficient_factor(1.5) > friction_coefficient_factor(3.0)

    def test_invalid_taper_ratio(self):
        with pytest.raises(ValueError):
            load_coefficient(1.0)
        with pytest.raises(ValueError):
            friction_coefficient_factor(0.5)


class TestThrustBearingConstruction:
    """Construction and validation."""

    def test_is_a_mecha_element(self):
        assert isinstance(make_thrust(), MechaElement)

    def test_sae_lubricant_path(self):
        from mecapy.bearings import viscosity

        bearing = make_thrust(viscosity=None, sae_grade=30, temperature=70.0)
        assert bearing.viscosity == pytest.approx(viscosity(30, 70.0))

    def test_lubricant_xor(self):
        with pytest.raises(ValueError):
            make_thrust(viscosity=None)
        with pytest.raises(ValueError):
            make_thrust(sae_grade=30, temperature=70.0)  # plus the default viscosity

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            make_thrust(inner_radius=0.0)
        with pytest.raises(ValueError):
            make_thrust(outer_radius=40.0)  # below the inner radius
        with pytest.raises(ValueError):
            make_thrust(n_pads=0)
        with pytest.raises(ValueError):
            make_thrust(n_pads=2.5)
        with pytest.raises(ValueError):
            make_thrust(taper_ratio=1.0)
        with pytest.raises(ValueError):
            make_thrust(pad_fraction=0.0)
        with pytest.raises(ValueError):
            make_thrust(pad_fraction=1.5)
        with pytest.raises(ValueError):
            make_thrust(speed=0.0)
        with pytest.raises(ValueError):
            make_thrust(load=-1.0)

    def test_repr(self):
        assert "ThrustBearing" in repr(make_thrust())


class TestThrustGeometry:
    """Pad geometry at the mean radius."""

    def test_mean_radius_and_pad_length(self):
        bearing = make_thrust()
        assert bearing.mean_radius == pytest.approx(75.0)
        assert bearing.pad_length == pytest.approx(50.0)

    def test_pad_width_shares_the_circumference(self):
        """Eight pads at 80% coverage split 2*pi*rm between them."""
        bearing = make_thrust()
        assert bearing.pad_width == pytest.approx(0.8 * 2.0 * math.pi * 75.0 / 8.0)

    def test_sliding_velocity(self):
        bearing = make_thrust()
        assert bearing.sliding_velocity == pytest.approx(
            2.0 * math.pi * 75.0 * 30.0 / 1000.0
        )

    def test_pad_load_and_pressure(self):
        bearing = make_thrust()
        assert bearing.pad_load == pytest.approx(2500.0)
        assert bearing.pressure == pytest.approx(
            bearing.pad_load / bearing.pad_area, rel=1e-12
        )


class TestThrustFilmSolution:
    """The closed-form Reynolds solution and its inverses."""

    def test_film_thickness_carries_the_load(self):
        """Feeding h2 back into the load formula returns the pad load."""
        bearing = make_thrust()
        h2 = bearing.film_thickness() / 1000.0  # m
        width = bearing.pad_width / 1000.0
        length = bearing.pad_length / 1000.0
        load = (
            load_coefficient(bearing.taper_ratio)
            * (bearing.viscosity * 1e-3)
            * bearing.sliding_velocity
            * width**2
            * length
            / h2**2
        )
        assert load == pytest.approx(bearing.pad_load, rel=1e-9)

    def test_film_scales_with_root_viscosity(self):
        """W ~ 1/h2^2, so h2 ~ sqrt(mu) at a fixed load."""
        thin = make_thrust(viscosity=30.0).film_thickness()
        thick = make_thrust(viscosity=120.0).film_thickness()
        assert thick / thin == pytest.approx(2.0, rel=1e-9)

    def test_heavier_load_thins_the_film(self):
        assert (
            make_thrust(load=40000.0).film_thickness() < make_thrust().film_thickness()
        )

    def test_pressure_profile_integrates_to_the_pad_load(self):
        """The 1-D Reynolds pressure and the closed-form load agree."""
        bearing = make_thrust()
        profile = bearing.pressure_profile(n=20001)
        total = 0.0
        xs, ps = profile["x"], profile["pressure"]
        for index in range(len(xs) - 1):  # trapezoid, MPa*mm
            total += 0.5 * (ps[index] + ps[index + 1]) * (xs[index + 1] - xs[index])
        # MPa*mm * mm = N
        assert total * bearing.pad_length == pytest.approx(bearing.pad_load, rel=1e-6)

    def test_pressure_vanishes_at_both_ends(self):
        profile = make_thrust().pressure_profile(n=51)
        assert profile["pressure"][0] == pytest.approx(0.0, abs=1e-9)
        assert profile["pressure"][-1] == pytest.approx(0.0, abs=1e-9)

    def test_pmax_exceeds_the_mean_pressure(self):
        bearing = make_thrust()
        performance = bearing.performance()
        assert performance["pmax"] > bearing.pressure

    def test_film_thickness_ratio_matches_the_taper(self):
        performance = make_thrust().performance()
        assert performance["inlet_film"] / performance["film_thickness"] == (
            pytest.approx(OPTIMUM_TAPER_RATIO)
        )

    def test_profile_thickness_spans_h1_to_h2(self):
        bearing = make_thrust()
        profile = bearing.pressure_profile(n=101)
        performance = bearing.performance()
        assert profile["film_thickness"][0] == pytest.approx(
            performance["inlet_film"], rel=1e-12
        )
        assert profile["film_thickness"][-1] == pytest.approx(
            performance["film_thickness"], rel=1e-12
        )

    def test_invalid_profile_resolution(self):
        with pytest.raises(ValueError):
            make_thrust().pressure_profile(n=1)


class TestThrustPerformance:
    """Friction, power and flow."""

    def test_friction_coefficient_is_small(self):
        """A hydrodynamic film gives f of order 1e-3."""
        performance = make_thrust().performance()
        assert 1e-4 < performance["friction_coefficient"] < 1e-1

    def test_power_loss_is_friction_times_velocity(self):
        bearing = make_thrust()
        performance = bearing.performance()
        assert performance["power_loss"] == pytest.approx(
            performance["friction_force"] * bearing.n_pads * bearing.sliding_velocity,
            rel=1e-12,
        )

    def test_friction_torque_acts_at_the_mean_radius(self):
        bearing = make_thrust()
        performance = bearing.performance()
        assert performance["friction_torque"] == pytest.approx(
            performance["friction_force"] * bearing.n_pads * bearing.mean_radius,
            rel=1e-12,
        )

    def test_optimum_taper_ratio_search(self):
        assert make_thrust().optimum_taper_ratio() == pytest.approx(
            OPTIMUM_TAPER_RATIO, rel=1e-4
        )

    def test_temperature_rise_is_independent_of_viscosity_and_speed(self):
        """Thickening the oil raises the heat and the flow by the same factor.

        H ~ sqrt(mu) and Q ~ h2 ~ sqrt(mu) at a fixed load, so the
        adiabatic rise cancels out to a function of pad pressure alone --
        the thrust analogue of the journal bearing's dT = 8.30 P (...).
        """
        base = make_thrust().temperature_rise()
        assert base > 0
        assert make_thrust(viscosity=120.0).temperature_rise() == pytest.approx(
            base, rel=1e-9
        )
        assert make_thrust(speed=60.0).temperature_rise() == pytest.approx(
            base, rel=1e-9
        )

    def test_temperature_rise_is_proportional_to_load(self):
        """What is left is the pad pressure: doubling the load doubles dT."""
        base = make_thrust().temperature_rise()
        assert make_thrust(load=40000.0).temperature_rise() == pytest.approx(
            2.0 * base, rel=1e-9
        )

    def test_temperature_rise_invalid_inputs(self):
        bearing = make_thrust()
        with pytest.raises(ValueError):
            bearing.temperature_rise(density=0.0)
        with pytest.raises(ValueError):
            bearing.temperature_rise(specific_heat=-1.0)


class TestThrustReport:
    """describe() and the plot."""

    def test_describe_returns_labelled_string(self):
        text = make_thrust(name="collar").describe()
        assert isinstance(text, str)
        assert "ThrustBearing geometry 'collar'" in text
        assert "mean radius (rm) = 75.000 mm" in text
        assert "pad count (n) = 8" in text

    def test_plot_returns_figure(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        figure = make_thrust().plot_pressure(show=False)
        assert isinstance(figure, Figure)
        matplotlib.pyplot.close("all")


class TestThrustPintInputs:
    """Optional pint quantities at the boundary."""

    def test_dimensions_accept_quantities(self):
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        quantity = make_thrust(
            inner_radius=5 * ureg.cm, load=20 * ureg.kN, speed=1800 * ureg.rpm
        )
        plain = make_thrust()
        assert quantity.film_thickness() == pytest.approx(
            plain.film_thickness(), rel=1e-12
        )
