"""Tests for the brake subsystem (shoe, band, caliper disc)."""

import math

import pytest

from mecapy import MechaElement
from mecapy.brakes import (BandBrake, CaliperDiscBrake, DiscBrake,
                           ExternalShoeBrake, InternalShoeBrake)
from mecapy.clutches import DiscClutch

# Shigley Ex. 16-2 geometry (Fig. 16-8): 300 mm drum, hinge pin at
# a = sqrt(112^2 + 50^2) from the drum center, force arm c = 212 mm.
SHIGLEY_A = math.sqrt(112 ** 2 + 50 ** 2)


def shigley_shoe(rotation="self_energizing"):
    """The Ex. 16-2 shoe: r=150, b=32, mu=0.32, theta 0..126 deg, pa=1 MPa."""
    return InternalShoeBrake(drum_radius=150, face_width=32,
                             pivot_distance=SHIGLEY_A, theta1=0, theta2=126,
                             actuation_arm=212, mu=0.32, rotation=rotation)


class TestShoeBrakeConstruction:
    def test_is_mecha_element(self):
        assert isinstance(shigley_shoe(), MechaElement)

    def test_theta_a_capped_at_90(self):
        assert shigley_shoe().theta_a == pytest.approx(90.0)
        short = InternalShoeBrake(150, 32, SHIGLEY_A, 10, 75, 212, mu=0.32)
        assert short.theta_a == pytest.approx(75.0)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            InternalShoeBrake(0, 32, SHIGLEY_A, 0, 126, 212)
        with pytest.raises(ValueError):
            InternalShoeBrake(150, 32, SHIGLEY_A, 126, 126, 212)  # theta1 == theta2
        with pytest.raises(ValueError):
            InternalShoeBrake(150, 32, SHIGLEY_A, 0, 200, 212)  # theta2 > 180
        with pytest.raises(ValueError):
            InternalShoeBrake(150, 32, SHIGLEY_A, 0, 126, 212, rotation="backwards")
        with pytest.raises(ValueError):
            shigley_shoe().torque(0.0)


class TestShigleyExample16_2:
    """Anchor: Shigley's Mechanical Engineering Design, Ex. 16-2."""

    def test_friction_moment(self):
        assert shigley_shoe().moment_friction(1.0) == pytest.approx(304e3, rel=1e-2)

    def test_normal_moment(self):
        assert shigley_shoe().moment_normal(1.0) == pytest.approx(788e3, rel=1e-2)

    def test_actuating_force(self):
        assert shigley_shoe().actuating_force(1.0) == pytest.approx(2280, rel=1e-2)

    def test_primary_shoe_torque(self):
        assert shigley_shoe().torque(1.0) == pytest.approx(366e3, rel=1e-2)

    def test_secondary_shoe_at_shared_force(self):
        # Both shoes see the same actuating force; the de-energizing shoe
        # runs at a lower peak pressure and delivers less torque.
        force = shigley_shoe().actuating_force(1.0)
        secondary = shigley_shoe(rotation="de_energizing")
        assert secondary.max_pressure_for_force(force) == pytest.approx(0.443, rel=1e-2)
        assert secondary.torque_for_force(force) == pytest.approx(162e3, rel=1e-2)

    def test_total_braking_capacity(self):
        force = shigley_shoe().actuating_force(1.0)
        total = (shigley_shoe().torque(1.0)
                 + shigley_shoe(rotation="de_energizing").torque_for_force(force))
        assert total == pytest.approx(528e3, rel=1e-2)


class TestSelfEnergizingAndLocking:
    def test_self_energizing_needs_less_force(self):
        primary = shigley_shoe()
        secondary = shigley_shoe(rotation="de_energizing")
        assert primary.actuating_force(1.0) < secondary.actuating_force(1.0)

    def test_self_locking_flips_with_mu(self):
        shoe = shigley_shoe()
        assert not shoe.is_self_locking
        assert shoe.self_locking_margin > 1
        # Raise mu until Mf exceeds MN: margin scales ~1/mu.
        mu_lock = 0.32 * shoe.self_locking_margin * 1.05
        locked = InternalShoeBrake(150, 32, SHIGLEY_A, 0, 126, 212, mu=mu_lock)
        assert locked.is_self_locking
        with pytest.raises(ValueError):
            locked.actuating_force(1.0)

    def test_de_energizing_never_self_locks(self):
        shoe = shigley_shoe(rotation="de_energizing")
        assert not shoe.is_self_locking

    def test_pressure_force_round_trip(self):
        shoe = shigley_shoe()
        force = shoe.actuating_force(0.7)
        assert shoe.max_pressure_for_force(force) == pytest.approx(0.7)


class TestHingeReactions:
    def test_reaction_components(self):
        shoe = shigley_shoe()
        k = 1.0 * 32 * 150 / 1.0  # pa * b * r / sin(theta_a)
        a_int = shoe._int_sin_cos
        b_int = shoe._int_sin2
        rx, ry = shoe.hinge_reactions(1.0)
        assert rx == pytest.approx(k * (a_int - 0.32 * b_int))
        assert ry == pytest.approx(k * (b_int + 0.32 * a_int))

    def test_applied_force_subtracts(self):
        shoe = shigley_shoe()
        rx0, ry0 = shoe.hinge_reactions(1.0)
        rx, ry = shoe.hinge_reactions(1.0, force_x=100.0, force_y=200.0)
        assert rx == pytest.approx(rx0 - 100.0)
        assert ry == pytest.approx(ry0 - 200.0)

    def test_de_energizing_flips_friction_terms(self):
        primary = shigley_shoe()
        secondary = shigley_shoe(rotation="de_energizing")
        rx_p, ry_p = primary.hinge_reactions(1.0)
        rx_s, ry_s = secondary.hinge_reactions(1.0)
        k = 1.0 * 32 * 150
        assert rx_s - rx_p == pytest.approx(2 * 0.32 * primary._int_sin2 * k)
        assert ry_p - ry_s == pytest.approx(2 * 0.32 * primary._int_sin_cos * k)


class TestExternalShoeBrake:
    def test_same_mechanics_as_internal(self):
        internal = shigley_shoe()
        external = ExternalShoeBrake(drum_radius=150, face_width=32,
                                     pivot_distance=SHIGLEY_A, theta1=0,
                                     theta2=126, actuation_arm=212, mu=0.32)
        assert external.torque(1.0) == pytest.approx(internal.torque(1.0))
        assert external.actuating_force(1.0) == pytest.approx(
            internal.actuating_force(1.0))


class TestBandBrake:
    def make_band(self):
        return BandBrake(drum_diameter=400, band_width=80, wrap_angle=270,
                         mu=0.30, band_thickness=3.0)

    def test_tension_ratio(self):
        # e^(0.30 * 3pi/2) ~ 4.111
        assert self.make_band().tension_ratio == pytest.approx(4.111, rel=1e-3)

    def test_torque(self):
        band = self.make_band()
        p1 = 5000.0
        p2 = band.slack_tension(p1)
        assert band.torque(p1) == pytest.approx((p1 - p2) * 200)

    def test_torque_tension_round_trip(self):
        band = self.make_band()
        torque = band.torque(5000.0)
        assert band.tight_tension_for_torque(torque) == pytest.approx(5000.0)

    def test_max_pressure(self):
        band = self.make_band()
        assert band.max_pressure(5000.0) == pytest.approx(2 * 5000 / (80 * 400))

    def test_pressure_tension_round_trip(self):
        band = self.make_band()
        p1 = band.tight_tension_for_pressure(0.5)
        assert band.max_pressure(p1) == pytest.approx(0.5)

    def test_band_stress_and_safety(self):
        band = self.make_band()
        assert band.band_stress(6000.0) == pytest.approx(6000 / (80 * 3))
        # steel Sy = 250 MPa
        assert band.band_safety_factor(6000.0) == pytest.approx(250 / 25.0)

    def test_band_stress_requires_thickness(self):
        band = BandBrake(400, 80, 270, mu=0.30)
        with pytest.raises(ValueError):
            band.band_stress(6000.0)

    def test_lining_pressure_safety_factor(self):
        band = BandBrake(400, 80, 270, lining="woven")
        p1 = band.tight_tension_for_pressure(0.26)
        assert band.pressure_safety_factor(p1) == pytest.approx(0.52 / 0.26)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            BandBrake(0, 80, 270)
        with pytest.raises(ValueError):
            BandBrake(400, 80, 0)
        with pytest.raises(ValueError):
            BandBrake(400, 80, 2000)  # more than 3 turns
        with pytest.raises(ValueError):
            self.make_band().torque(0.0)


class TestCaliperDiscBrake:
    def make_brake(self):
        return CaliperDiscBrake(outer_radius=150, inner_radius=100,
                                pad_angle=60, mu=0.35, n_pads=2)

    def test_effective_radii(self):
        brake = self.make_brake()
        assert brake.effective_radius_uniform_wear == pytest.approx(125.0)
        expected = 2 * (150 ** 3 - 100 ** 3) / (3 * (150 ** 2 - 100 ** 2))
        assert brake.effective_radius_uniform_pressure == pytest.approx(expected)
        assert (brake.effective_radius_uniform_pressure
                > brake.effective_radius_uniform_wear)

    def test_uniform_wear_force_and_torque(self):
        brake = self.make_brake()
        theta_p = math.radians(60)
        force = brake.actuating_force_uniform_wear(1.0)
        assert force == pytest.approx(theta_p * 1.0 * 100 * 50)
        assert brake.torque_uniform_wear(force) == pytest.approx(
            2 * 0.35 * force * 125.0)

    def test_uniform_wear_pressure_round_trip(self):
        brake = self.make_brake()
        force = brake.actuating_force_uniform_wear(0.8)
        assert brake.max_pressure_uniform_wear(force) == pytest.approx(0.8)

    def test_uniform_pressure_round_trip(self):
        brake = self.make_brake()
        force = brake.actuating_force_uniform_pressure(0.8)
        assert brake.pressure_uniform_pressure(force) == pytest.approx(0.8)
        assert force == pytest.approx(0.8 * brake.pad_area)

    def test_uniform_pressure_torque_exceeds_uniform_wear(self):
        brake = self.make_brake()
        assert (brake.torque_uniform_pressure(5000)
                > brake.torque_uniform_wear(5000))

    def test_pressure_safety_factor_with_lining(self):
        brake = CaliperDiscBrake(150, 100, 60, lining="sintered_metal")
        force = brake.actuating_force_uniform_wear(2.07 / 3)
        assert brake.pressure_safety_factor(force) == pytest.approx(3.0)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            CaliperDiscBrake(100, 150, 60)  # inverted radii
        with pytest.raises(ValueError):
            CaliperDiscBrake(150, 100, 0)
        with pytest.raises(ValueError):
            CaliperDiscBrake(150, 100, 60, n_pads=0)
        with pytest.raises(ValueError):
            self.make_brake().torque_uniform_wear(0.0)


class TestDiscBrakeAlias:
    def test_alias_is_disc_clutch(self):
        assert DiscBrake is DiscClutch
