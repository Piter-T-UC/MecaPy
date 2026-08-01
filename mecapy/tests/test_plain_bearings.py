"""Tests for boundary-lubricated plain (bushing) bearings."""

import math

import pytest
from mecapy import MechaElement
from mecapy.bearings import BUSHING_MATERIALS, PlainBearing, get_bushing_material


def make_bushing(**kwargs):
    """Bushing with clean hand numbers.

    d = 25 mm, l = 25 mm, W = 1000 N, N = 5 rev/s:
    P = 1000/625 = 1.6 MPa, V = pi*25*5/1000 = 0.3927 m/s,
    PV = 0.6283 MPa*m/s.
    """
    defaults = dict(
        bore_diameter=25.0,
        length=25.0,
        load=1000.0,
        speed=5.0,
        bushing_material="cast_bronze",
    )
    defaults.update(kwargs)
    return PlainBearing(**defaults)


class TestBushingData:
    """The Shigley Table 12-8 accessor."""

    def test_lookup_returns_limits(self):
        row = get_bushing_material("cast_bronze")
        assert row["p_max"] == pytest.approx(31.0)
        assert row["pv_max"] == pytest.approx(1.75)
        assert row["t_max"] == pytest.approx(163.0)

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError):
            get_bushing_material("unobtainium")

    def test_every_row_has_the_same_fields(self):
        fields = {"p_max", "v_max", "pv_max", "t_max", "mu"}
        for row in BUSHING_MATERIALS.values():
            assert set(row) == fields
            assert all(value > 0 for value in row.values())


class TestPlainBearingConstruction:
    """Construction, validation and the material-vs-explicit idiom."""

    def test_is_a_mecha_element(self):
        assert isinstance(make_bushing(), MechaElement)

    def test_material_supplies_the_friction_coefficient(self):
        assert make_bushing().mu == pytest.approx(
            BUSHING_MATERIALS["cast_bronze"]["mu"]
        )

    def test_explicit_mu_beats_the_table(self):
        """An explicit value always wins over the liner default."""
        assert make_bushing(mu=0.3).mu == pytest.approx(0.3)

    def test_default_mu_without_a_material(self):
        bearing = make_bushing(bushing_material=None)
        assert 0 < bearing.mu < 1.5
        assert bearing.bushing_material is None

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            make_bushing(bore_diameter=0.0)
        with pytest.raises(ValueError):
            make_bushing(length=-1.0)
        with pytest.raises(ValueError):
            make_bushing(load=0.0)
        with pytest.raises(ValueError):
            make_bushing(speed=-2.0)
        with pytest.raises(ValueError):
            make_bushing(mu=0.0)
        with pytest.raises(ValueError):
            make_bushing(mu=2.0)
        with pytest.raises(ValueError):
            make_bushing(bushing_material="unobtainium")

    def test_setters_revalidate(self):
        bearing = make_bushing()
        bearing.load = 2000.0
        assert bearing.load == 2000.0
        with pytest.raises(ValueError):
            bearing.speed = 0.0

    def test_repr(self):
        assert "PlainBearing" in repr(make_bushing())


class TestPlainBearingOperatingPoint:
    """P, V and PV (Shigley sec. 12-15)."""

    def test_pressure_hand_check(self):
        """P = W / (d * l) = 1000 / 625 = 1.6 MPa."""
        assert make_bushing().pressure == pytest.approx(1.6)

    def test_velocity_hand_check(self):
        """V = pi * d * N = pi * 25 * 5 / 1000 m/s."""
        assert make_bushing().rubbing_velocity == pytest.approx(
            math.pi * 25.0 * 5.0 / 1000.0
        )

    def test_pv_is_the_product(self):
        bearing = make_bushing()
        assert bearing.pv == pytest.approx(
            bearing.pressure * bearing.rubbing_velocity, rel=1e-12
        )
        assert bearing.pv == pytest.approx(0.62832, rel=1e-4)

    def test_friction_and_power(self):
        bearing = make_bushing()
        assert bearing.friction_force() == pytest.approx(bearing.mu * 1000.0)
        assert bearing.friction_torque() == pytest.approx(bearing.mu * 1000.0 * 12.5)
        assert bearing.power_loss() == pytest.approx(
            bearing.mu * 1000.0 * bearing.rubbing_velocity, rel=1e-12
        )

    def test_heat_flux_is_mu_times_pv(self):
        """H/A and mu*PV are the same quantity in different units."""
        bearing = make_bushing()
        assert bearing.heat_flux() == pytest.approx(bearing.mu * bearing.pv, rel=1e-12)

    def test_l_over_d_and_speed_rpm(self):
        bearing = make_bushing()
        assert bearing.l_over_d == pytest.approx(1.0)
        assert bearing.speed_rpm == pytest.approx(300.0)


class TestPlainBearingLimits:
    """Pressure, velocity and PV checks against the liner."""

    def test_safety_factors_are_allowable_over_actual(self):
        bearing = make_bushing()
        row = BUSHING_MATERIALS["cast_bronze"]
        assert bearing.pressure_safety_factor() == pytest.approx(
            row["p_max"] / bearing.pressure, rel=1e-12
        )
        assert bearing.velocity_safety_factor() == pytest.approx(
            row["v_max"] / bearing.rubbing_velocity, rel=1e-12
        )
        assert bearing.pv_safety_factor() == pytest.approx(
            row["pv_max"] / bearing.pv, rel=1e-12
        )

    def test_pv_check_all_pass(self):
        result = make_bushing().pv_check(temperature=80.0)
        assert result == {
            "pressure": True,
            "velocity": True,
            "pv": True,
            "temperature": True,
        }

    def test_pv_check_skips_absent_temperature(self):
        assert make_bushing().pv_check()["temperature"] is None

    def test_pv_binds_before_pressure_and_velocity(self):
        """A PTFE bushing can pass P and V and still fail PV.

        P = 1875/625 = 3.0 MPa (limit 3.45), V = 0.2 m/s (limit 0.25),
        but PV = 0.6 MPa*m/s against a limit of 0.035.
        """
        speed = 0.2 * 1000.0 / (math.pi * 25.0)  # rev/s giving V = 0.2 m/s
        bearing = make_bushing(bushing_material="ptfe", load=1875.0, speed=speed)
        assert bearing.pressure == pytest.approx(3.0)
        assert bearing.rubbing_velocity == pytest.approx(0.2)
        result = bearing.pv_check()
        assert result["pressure"] is True
        assert result["velocity"] is True
        assert result["pv"] is False
        assert bearing.pv_safety_factor() < 1.0

    def test_temperature_over_limit_is_flagged(self):
        assert make_bushing().pv_check(temperature=200.0)["temperature"] is False

    def test_maximum_load_and_speed_are_consistent(self):
        """At the limiting load the binding safety factor is exactly 1."""
        bearing = make_bushing()
        at_limit = make_bushing(load=bearing.maximum_load())
        assert min(
            at_limit.pressure_safety_factor(), at_limit.pv_safety_factor()
        ) == pytest.approx(1.0, rel=1e-9)
        faster = make_bushing(speed=bearing.maximum_speed())
        assert min(
            faster.velocity_safety_factor(), faster.pv_safety_factor()
        ) == pytest.approx(1.0, rel=1e-9)

    def test_checks_need_a_material(self):
        bearing = make_bushing(bushing_material=None)
        for method in (
            bearing.pressure_safety_factor,
            bearing.velocity_safety_factor,
            bearing.pv_safety_factor,
            bearing.pv_check,
            bearing.maximum_load,
            bearing.maximum_speed,
        ):
            with pytest.raises(ValueError):
                method()


class TestPlainBearingWear:
    """Linear wear model."""

    def test_wear_round_trip(self):
        """hours_to_wear inverts wear_depth exactly."""
        bearing = make_bushing()
        hours = bearing.hours_to_wear(0.1, wear_factor=1e-8)
        assert bearing.wear_depth(hours, wear_factor=1e-8) == pytest.approx(
            0.1, rel=1e-9
        )

    def test_wear_grows_with_pv(self):
        light = make_bushing(load=500.0).wear_depth(1000.0, 1e-8)
        heavy = make_bushing(load=2000.0).wear_depth(1000.0, 1e-8)
        assert heavy > light

    def test_invalid_inputs(self):
        bearing = make_bushing()
        with pytest.raises(ValueError):
            bearing.wear_depth(-1.0, 1e-8)
        with pytest.raises(ValueError):
            bearing.wear_depth(100.0, 0.0)
        with pytest.raises(ValueError):
            bearing.hours_to_wear(0.0, 1e-8)
        with pytest.raises(ValueError):
            bearing.hours_to_wear(0.1, -1.0)


class TestPlainBearingReport:
    """describe()."""

    def test_describe_returns_labelled_string(self):
        text = make_bushing(name="idler").describe()
        assert isinstance(text, str)
        assert "PlainBearing geometry 'idler'" in text
        assert "bore diameter (d) = 25.000 mm" in text
        assert "severity index (PV) = 0.6283 MPa*m/s" in text
        assert "liner = cast_bronze" in text

    def test_describe_without_a_material(self):
        text = make_bushing(bushing_material=None).describe()
        assert "liner = not given (PV checks unavailable)" in text


class TestPlainBearingPintInputs:
    """Optional pint quantities at the boundary."""

    def test_dimensions_accept_quantities(self):
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        quantity = make_bushing(
            bore_diameter=1 * ureg.inch, load=1 * ureg.kN, speed=300 * ureg.rpm
        )
        plain = make_bushing(bore_diameter=25.4, load=1000.0, speed=5.0)
        assert quantity.pv == pytest.approx(plain.pv, rel=1e-12)

    def test_wrong_dimension_is_rejected(self):
        pint = pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        with pytest.raises(pint.DimensionalityError):
            make_bushing(bore_diameter=5 * ureg.newton)
