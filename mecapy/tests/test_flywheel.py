"""Tests for the Flywheel class and the Wheel inertia/energy additions."""

import math

import pytest

from mecapy import MechaElement
from mecapy.wheels import Flywheel, Wheel

STEEL_RHO = 7850.0
STEEL_NU = 0.3
STEEL_SY = 250e6


def example_flywheel():
    """Solid steel disc, ro = 0.3 m, t = 0.05 m."""
    return Flywheel(outer_radius=0.3, thickness=0.05)


class TestWheelAdditions:
    def test_moment_of_inertia(self):
        wheel = Wheel(radius=0.5, mass=40.0)
        assert wheel.moment_of_inertia == pytest.approx(0.5 * 40 * 0.25)

    def test_kinetic_energy(self):
        wheel = Wheel(radius=0.5, mass=40.0)
        assert wheel.kinetic_energy(10.0) == pytest.approx(
            0.5 * wheel.moment_of_inertia * 100)

    def test_inertia_requires_mass(self):
        with pytest.raises(ValueError):
            Wheel(radius=0.5).moment_of_inertia


class TestFlywheelConstruction:
    def test_is_wheel_and_mecha_element(self):
        flywheel = example_flywheel()
        assert isinstance(flywheel, Wheel)
        assert isinstance(flywheel, MechaElement)

    def test_mass_from_geometry(self):
        flywheel = example_flywheel()
        assert flywheel.mass == pytest.approx(
            STEEL_RHO * math.pi * 0.3 ** 2 * 0.05)

    def test_annular_mass(self):
        flywheel = Flywheel(outer_radius=0.3, inner_radius=0.1, thickness=0.05)
        assert flywheel.mass == pytest.approx(
            STEEL_RHO * math.pi * (0.3 ** 2 - 0.1 ** 2) * 0.05)

    def test_explicit_mass(self):
        assert Flywheel(outer_radius=0.3, mass=25.0).mass == 25.0

    def test_mass_recomputes_when_radius_changes(self):
        """Built from thickness, mass and inertia track a radius change."""
        flywheel = Flywheel(outer_radius=0.3, thickness=0.05)
        m0 = flywheel.mass
        flywheel.radius = 0.4  # outer_radius alias
        assert flywheel.mass == pytest.approx(m0 * (0.4 ** 2) / (0.3 ** 2))
        assert flywheel.moment_of_inertia == pytest.approx(
            0.5 * flywheel.mass * 0.4 ** 2)

    def test_mass_recomputes_when_thickness_changes(self):
        """Thickness is settable and drives mass linearly."""
        flywheel = Flywheel(outer_radius=0.3, thickness=0.05)
        m0 = flywheel.mass
        flywheel.thickness = 0.10
        assert flywheel.mass == pytest.approx(2 * m0)
        with pytest.raises(ValueError):
            flywheel.thickness = -1

    def test_exactly_one_of_thickness_or_mass(self):
        with pytest.raises(ValueError):
            Flywheel(outer_radius=0.3)
        with pytest.raises(ValueError):
            Flywheel(outer_radius=0.3, thickness=0.05, mass=25.0)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            Flywheel(outer_radius=0, thickness=0.05)
        with pytest.raises(ValueError):
            Flywheel(outer_radius=0.3, inner_radius=0.3, thickness=0.05)
        with pytest.raises(ValueError):
            Flywheel(outer_radius=0.3, inner_radius=-0.1, thickness=0.05)
        with pytest.raises(ValueError):
            Flywheel(outer_radius=0.3, thickness=-0.05)


class TestFlywheelInertia:
    def test_solid_disc_inertia(self):
        flywheel = example_flywheel()
        assert flywheel.moment_of_inertia == pytest.approx(
            0.5 * flywheel.mass * 0.3 ** 2)

    def test_annular_inertia(self):
        flywheel = Flywheel(outer_radius=0.3, inner_radius=0.2, mass=30.0)
        assert flywheel.moment_of_inertia == pytest.approx(
            0.5 * 30 * (0.3 ** 2 + 0.2 ** 2))

    def test_thin_rim_limit(self):
        # ri -> ro: I -> m * r^2
        flywheel = Flywheel(outer_radius=0.3, inner_radius=0.299, mass=10.0)
        assert flywheel.moment_of_inertia == pytest.approx(10 * 0.3 ** 2, rel=1e-2)


class TestEnergyFluctuation:
    def test_required_inertia(self):
        # Ue = 1500 J, Cs = 0.05, w = 104.7 rad/s -> I ~ 2.737 kg*m^2
        assert Flywheel.required_inertia(1500, 0.05, 104.7) == pytest.approx(
            2.737, rel=1e-3)

    def test_energy_fluctuation(self):
        flywheel = example_flywheel()
        assert flywheel.energy_fluctuation(105.0, 95.0) == pytest.approx(
            0.5 * flywheel.moment_of_inertia * (105 ** 2 - 95 ** 2))

    def test_coefficient_of_fluctuation(self):
        assert Flywheel.coefficient_of_fluctuation(105.0, 95.0) == pytest.approx(
            10 / 100)

    def test_sizing_round_trip(self):
        # A flywheel with exactly the required inertia reproduces the target Cs.
        inertia = Flywheel.required_inertia(1500, 0.05, 100.0)
        flywheel = Flywheel(outer_radius=0.3, mass=inertia / (0.5 * 0.3 ** 2))
        omega_max, omega_min = flywheel.speed_swing(1500, 100.0)
        assert Flywheel.coefficient_of_fluctuation(omega_max, omega_min) == \
            pytest.approx(0.05)

    def test_speed_swing_too_large(self):
        with pytest.raises(ValueError):
            example_flywheel().speed_swing(1e9, 10.0)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            Flywheel.required_inertia(0, 0.05, 100.0)
        with pytest.raises(ValueError):
            Flywheel.required_inertia(1500, 0, 100.0)
        with pytest.raises(ValueError):
            example_flywheel().energy_fluctuation(95.0, 105.0)


class TestRotatingDiscStresses:
    def test_solid_disc_peak_tangential(self):
        # sigma_t,max = (3+nu)/8 * rho * w^2 * ro^2 at the center
        flywheel = example_flywheel()
        omega = 300.0
        expected = (3 + STEEL_NU) / 8 * STEEL_RHO * omega ** 2 * 0.3 ** 2
        assert flywheel.tangential_stress(omega) == pytest.approx(expected)
        assert flywheel.radial_stress(omega) == pytest.approx(expected)

    def test_radial_stress_zero_at_free_surfaces(self):
        solid = example_flywheel()
        assert solid.radial_stress(300.0, radius=0.3) == pytest.approx(0.0, abs=1.0)
        annular = Flywheel(outer_radius=0.3, inner_radius=0.1, thickness=0.05)
        assert annular.radial_stress(300.0, radius=0.1) == pytest.approx(0.0, abs=1.0)
        assert annular.radial_stress(300.0, radius=0.3) == pytest.approx(0.0, abs=1.0)

    def test_annular_peak_at_bore(self):
        flywheel = Flywheel(outer_radius=0.3, inner_radius=0.1, thickness=0.05)
        peak = flywheel.tangential_stress(300.0)  # defaults to the bore
        assert peak == pytest.approx(flywheel.tangential_stress(300.0, radius=0.1))
        assert peak > flywheel.tangential_stress(300.0, radius=0.2)

    def test_thin_rim_approaches_hoop_formula(self):
        # A thin rim's full-solution peak approaches rho*w^2*r^2.
        flywheel = Flywheel(outer_radius=0.3, inner_radius=0.29, mass=10.0)
        omega = 300.0
        full = flywheel.tangential_stress(omega)
        hoop = flywheel.rim_hoop_stress(omega)
        assert full == pytest.approx(hoop, rel=0.05)

    def test_burst_safety_factor_and_max_speed_round_trip(self):
        flywheel = example_flywheel()
        omega_max = flywheel.max_speed(safety_factor=1.0)
        assert flywheel.burst_safety_factor(omega_max) == pytest.approx(1.0)
        omega_2 = flywheel.max_speed(safety_factor=2.0)
        assert flywheel.burst_safety_factor(omega_2) == pytest.approx(2.0)
        assert flywheel.max_speed_rpm() == pytest.approx(
            omega_max * 60 / (2 * math.pi))

    def test_stress_scales_with_omega_squared(self):
        flywheel = example_flywheel()
        assert flywheel.tangential_stress(600.0) == pytest.approx(
            4 * flywheel.tangential_stress(300.0))

    def test_radius_out_of_range(self):
        flywheel = Flywheel(outer_radius=0.3, inner_radius=0.1, thickness=0.05)
        with pytest.raises(ValueError):
            flywheel.tangential_stress(300.0, radius=0.05)
        with pytest.raises(ValueError):
            flywheel.tangential_stress(300.0, radius=0.4)


class TestFlywheelEnergy:
    def test_kinetic_energy_inherited(self):
        flywheel = example_flywheel()
        assert flywheel.kinetic_energy(100.0) == pytest.approx(
            0.5 * flywheel.moment_of_inertia * 100 ** 2)

    def test_repr(self):
        assert "Flywheel" in repr(example_flywheel())
