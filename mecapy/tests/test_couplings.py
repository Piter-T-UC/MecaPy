"""Tests for the coupling subsystem (flange, flexible)."""

import math

import pytest

from mecapy import MechaElement
from mecapy.couplings import FlangeCoupling, FlexibleCoupling


def example_flange():
    """Hand-solved case: T = 500 N*m, 4 bolts on a 200 mm circle.

    F_bolt = 2*500e3/(4*200) = 1250 N; with 10 mm bolts
    tau = 1250/(pi*25) ~ 15.9 MPa.
    """
    return FlangeCoupling(shaft_diameter=50, bolt_circle_diameter=200,
                          n_bolts=4, bolt_diameter=10, flange_thickness=15,
                          hub_diameter=100, key_width=14, key_height=9,
                          key_length=70)


class TestFlangeCouplingConstruction:
    def test_is_mecha_element(self):
        assert isinstance(example_flange(), MechaElement)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            FlangeCoupling(0, 200, 4, 10, 15)
        with pytest.raises(ValueError):
            FlangeCoupling(50, 200, 1, 10, 15)  # too few bolts
        with pytest.raises(ValueError):
            FlangeCoupling(250, 200, 4, 10, 15)  # bolt circle inside shaft
        with pytest.raises(ValueError):
            FlangeCoupling(50, 200, 4, 10, 15, hub_diameter=40)  # hub < shaft
        with pytest.raises(ValueError):
            FlangeCoupling(50, 200, 4, 10, 15, hub_diameter=220)  # hub > circle
        with pytest.raises(ValueError):
            FlangeCoupling(50, 200, 4, 10, 15, key_width=14)  # partial key

    def test_key_args_all_or_none(self):
        no_key = FlangeCoupling(50, 200, 4, 10, 15)
        with pytest.raises(ValueError):
            no_key.key_shear_stress(500e3)


class TestFlangeCouplingBolts:
    def test_bolt_force(self):
        assert example_flange().bolt_force(500e3) == pytest.approx(1250.0)

    def test_bolt_shear_stress(self):
        coupling = example_flange()
        assert coupling.bolt_shear_stress(500e3) == pytest.approx(
            1250 / (math.pi * 25), rel=1e-6)
        assert coupling.bolt_shear_stress(500e3) == pytest.approx(15.92, rel=1e-3)

    def test_bolt_safety_factor(self):
        coupling = example_flange()
        # steel bolts: 0.577 * 250 MPa / 15.92 MPa
        assert coupling.bolt_safety_factor(500e3) == pytest.approx(
            0.577 * 250 / coupling.bolt_shear_stress(500e3))

    def test_flange_bearing_stress(self):
        assert example_flange().flange_bearing_stress(500e3) == pytest.approx(
            1250 / (10 * 15))


class TestFlangeCouplingHubAndKey:
    def test_hub_flange_shear_stress(self):
        coupling = example_flange()
        assert coupling.hub_flange_shear_stress(500e3) == pytest.approx(
            2 * 500e3 / (math.pi * 100 ** 2 * 15))

    def test_hub_check_requires_hub_diameter(self):
        no_hub = FlangeCoupling(50, 200, 4, 10, 15)
        with pytest.raises(ValueError):
            no_hub.hub_flange_shear_stress(500e3)

    def test_key_stresses(self):
        coupling = example_flange()
        key_force = 2 * 500e3 / 50
        assert coupling.key_force(500e3) == pytest.approx(key_force)
        assert coupling.key_shear_stress(500e3) == pytest.approx(
            key_force / (14 * 70))
        assert coupling.key_bearing_stress(500e3) == pytest.approx(
            2 * key_force / (9 * 70))

    def test_key_safety_factor_is_min_of_modes(self):
        coupling = example_flange()
        sy = 180.0  # cast iron, MPa
        sf_shear = 0.577 * sy / coupling.key_shear_stress(500e3)
        sf_bearing = sy / coupling.key_bearing_stress(500e3)
        assert coupling.key_safety_factor(500e3) == pytest.approx(
            min(sf_shear, sf_bearing))


class TestFlangeCouplingCapacity:
    def test_capacity_respects_safety_factor(self):
        coupling = example_flange()
        torque = coupling.torque_capacity(safety_factor=2.0)
        # At the capacity torque every mode has a safety factor >= 2.
        assert coupling.bolt_safety_factor(torque) >= 2.0 - 1e-9
        assert coupling.flange_bearing_safety_factor(torque) >= 2.0 - 1e-9
        assert coupling.hub_flange_safety_factor(torque) >= 2.0 - 1e-9
        assert coupling.key_safety_factor(torque) >= 2.0 - 1e-9
        # And the weakest mode is exactly at 2.
        weakest = min(coupling.bolt_safety_factor(torque),
                      coupling.flange_bearing_safety_factor(torque),
                      coupling.hub_flange_safety_factor(torque),
                      coupling.key_safety_factor(torque))
        assert weakest == pytest.approx(2.0)

    def test_capacity_skips_missing_modes(self):
        bare = FlangeCoupling(50, 200, 4, 10, 15)
        assert bare.torque_capacity() > 0

    def test_invalid_safety_factor(self):
        with pytest.raises(ValueError):
            example_flange().torque_capacity(safety_factor=0)


class TestFlexibleCoupling:
    def make_coupling(self):
        return FlexibleCoupling(torque_rating=250e3, max_speed_rpm=3600,
                                max_angular_misalignment=1.0,
                                max_parallel_offset=0.4, max_axial_movement=1.5,
                                torsional_stiffness=2e6)

    def test_is_mecha_element(self):
        assert isinstance(self.make_coupling(), MechaElement)

    def test_service_torque(self):
        assert FlexibleCoupling.service_torque(100e3, 1.75) == pytest.approx(175e3)

    def test_torque_and_speed_safety_factors(self):
        coupling = self.make_coupling()
        assert coupling.torque_safety_factor(125e3) == pytest.approx(2.0)
        assert coupling.speed_safety_factor(1800) == pytest.approx(2.0)

    def test_check_misalignment(self):
        coupling = self.make_coupling()
        assert coupling.check_misalignment(angular=0.5, parallel=0.2, axial=1.0)
        assert not coupling.check_misalignment(angular=1.5)
        assert not coupling.check_misalignment(parallel=0.5)
        assert not coupling.check_misalignment(axial=2.0)

    def test_validate_misalignment_names_violation(self):
        coupling = self.make_coupling()
        coupling.validate_misalignment(angular=0.5)
        with pytest.raises(ValueError, match="Angular"):
            coupling.validate_misalignment(angular=1.5)
        with pytest.raises(ValueError, match="Parallel"):
            coupling.validate_misalignment(parallel=0.5)
        with pytest.raises(ValueError, match="Axial"):
            coupling.validate_misalignment(axial=2.0)

    def test_windup(self):
        coupling = self.make_coupling()
        assert coupling.torsional_deflection(100e3) == pytest.approx(0.05)
        assert coupling.windup_angle_degrees(100e3) == pytest.approx(
            math.degrees(0.05))

    def test_windup_requires_stiffness(self):
        coupling = FlexibleCoupling(250e3, 3600, 1.0, 0.4, 1.5)
        with pytest.raises(ValueError):
            coupling.torsional_deflection(100e3)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            FlexibleCoupling(0, 3600, 1.0, 0.4, 1.5)
        with pytest.raises(ValueError):
            FlexibleCoupling(250e3, 0, 1.0, 0.4, 1.5)
        with pytest.raises(ValueError):
            FlexibleCoupling(250e3, 3600, -1.0, 0.4, 1.5)
        with pytest.raises(ValueError):
            self.make_coupling().check_misalignment(angular=-1)
