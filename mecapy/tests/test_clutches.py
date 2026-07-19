"""Tests for the clutch subsystem (disc, cone, centrifugal, friction data)."""

import math

import pytest

from mecapy import MechaElement
from mecapy.clutches import (AxialFrictionInterface, CentrifugalClutch,
                             ConeClutch, DiscClutch, get_friction_material)


def example_disc_clutch():
    """Disc clutch with a closed-form hand solution.

    D = 300 mm, d = 225 mm, mu = 0.25, 2 faces:
    uniform wear at pa = 0.828 MPa -> F ~ 21.95 kN, T ~ 1.44e6 N*mm.
    """
    return DiscClutch(outer_diameter=300, inner_diameter=225, mu=0.25, n_faces=2)


class TestFrictionData:
    def test_known_lining(self):
        molded = get_friction_material("molded")
        assert molded["mu_dry"] == pytest.approx(0.35)
        assert molded["p_max"] == pytest.approx(1.55)
        assert molded["t_max"] == 260

    def test_oil_value_may_be_none(self):
        assert get_friction_material("cermet")["mu_oil"] is None

    def test_unknown_lining(self):
        with pytest.raises(ValueError):
            get_friction_material("unobtainium")


class TestDiscClutchConstruction:
    def test_is_mecha_element(self):
        assert isinstance(example_disc_clutch(), MechaElement)

    def test_lining_supplies_mu(self):
        clutch = DiscClutch(300, 225, lining="molded")
        assert clutch.mu == pytest.approx(0.35)

    def test_explicit_mu_wins_over_lining(self):
        clutch = DiscClutch(300, 225, mu=0.2, lining="molded")
        assert clutch.mu == pytest.approx(0.2)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            DiscClutch(0, 225)
        with pytest.raises(ValueError):
            DiscClutch(300, 0)
        with pytest.raises(ValueError):
            DiscClutch(225, 300)  # inverted diameters
        with pytest.raises(ValueError):
            DiscClutch(300, 225, n_faces=0)
        with pytest.raises(ValueError):
            DiscClutch(300, 225, mu=2.0)
        with pytest.raises(ValueError):
            DiscClutch(300, 225, lining="unobtainium")


class TestDiscClutchUniformWear:
    def test_actuating_force(self):
        # F = pi * pa * d * (D - d) / 2
        clutch = example_disc_clutch()
        force = clutch.actuating_force_uniform_wear(0.828)
        assert force == pytest.approx(math.pi * 0.828 * 225 * 75 / 2)
        assert force == pytest.approx(21948, rel=1e-3)

    def test_torque(self):
        clutch = example_disc_clutch()
        force = clutch.actuating_force_uniform_wear(0.828)
        torque = clutch.torque_uniform_wear(force)
        assert torque == pytest.approx(2 * 0.25 * force * 525 / 4)
        assert torque == pytest.approx(1.440e6, rel=1e-3)

    def test_torque_capacity_composes(self):
        clutch = example_disc_clutch()
        force = clutch.actuating_force_uniform_wear(0.828)
        assert clutch.torque_capacity_uniform_wear(0.828) == pytest.approx(
            clutch.torque_uniform_wear(force))

    def test_pressure_round_trip(self):
        clutch = example_disc_clutch()
        force = clutch.actuating_force_uniform_wear(0.828)
        assert clutch.max_pressure_uniform_wear(force) == pytest.approx(0.828)


class TestDiscClutchUniformPressure:
    def test_pressure_round_trip(self):
        clutch = example_disc_clutch()
        force = clutch.actuating_force_uniform_pressure(0.828)
        assert clutch.pressure_uniform_pressure(force) == pytest.approx(0.828)

    def test_uniform_pressure_torque_exceeds_uniform_wear(self):
        # The uniform-pressure effective radius is always the larger one.
        clutch = example_disc_clutch()
        assert (clutch.torque_uniform_pressure(10000)
                > clutch.torque_uniform_wear(10000))


class TestDiscClutchOptimum:
    def test_optimal_inner_diameter(self):
        clutch = example_disc_clutch()
        assert clutch.optimal_inner_diameter == pytest.approx(300 / math.sqrt(3))

    def test_is_optimal_ratio(self):
        optimal = DiscClutch(300, 300 / math.sqrt(3), mu=0.25)
        assert optimal.is_optimal_ratio()
        assert not example_disc_clutch().is_optimal_ratio()

    def test_optimum_maximizes_torque_capacity(self):
        d_star = 300 / math.sqrt(3)
        t_star = DiscClutch(300, d_star).torque_capacity_uniform_wear(1.0)
        for d in (0.9 * d_star, 1.1 * d_star):
            assert DiscClutch(300, d).torque_capacity_uniform_wear(1.0) < t_star


class TestDiscClutchChecks:
    def test_pressure_safety_factor_with_lining(self):
        clutch = DiscClutch(300, 225, lining="molded")
        force = clutch.actuating_force_uniform_wear(1.55 / 2)
        assert clutch.pressure_safety_factor(force) == pytest.approx(2.0)

    def test_pressure_safety_factor_requires_allowable(self):
        with pytest.raises(ValueError):
            example_disc_clutch().pressure_safety_factor(10000)

    def test_power(self):
        clutch = example_disc_clutch()
        torque_nm = clutch.torque_uniform_wear(10000) / 1e3
        omega = 1800 * 2 * math.pi / 60
        assert clutch.power(10000, 1800) == pytest.approx(torque_nm * omega)

    def test_invalid_theory(self):
        clutch = DiscClutch(300, 225, lining="molded")
        with pytest.raises(ValueError):
            clutch.pressure_safety_factor(10000, theory="magic")


class TestConeClutch:
    def test_flat_cone_matches_disc(self):
        # alpha = 90 degrees is exactly a flat disc; ConeClutch caps at 45,
        # so compare through the shared base class.
        flat = AxialFrictionInterface(300, 225, mu=0.25, cone_angle=90.0)
        disc = DiscClutch(300, 225, mu=0.25)
        assert flat.torque_uniform_wear(1000) == pytest.approx(
            disc.torque_uniform_wear(1000), rel=1e-12)
        assert flat.torque_uniform_pressure(1000) == pytest.approx(
            disc.torque_uniform_pressure(1000), rel=1e-12)

    def test_wedging_amplifies_torque(self):
        cone = ConeClutch(300, 225, cone_angle=12, mu=0.25)
        disc = DiscClutch(300, 225, mu=0.25)
        ratio = cone.torque_uniform_wear(1000) / disc.torque_uniform_wear(1000)
        assert ratio == pytest.approx(1 / math.sin(math.radians(12)))

    def test_face_width(self):
        cone = ConeClutch(300, 225, cone_angle=12)
        assert cone.face_width == pytest.approx(
            75 / (2 * math.sin(math.radians(12))))

    def test_self_holding(self):
        # tan(8 deg) ~ 0.14 < mu = 0.3 -> self-holding;
        # tan(30 deg) ~ 0.58 > 0.3 -> free release.
        assert ConeClutch(300, 225, cone_angle=8, mu=0.3).is_self_holding
        assert not ConeClutch(300, 225, cone_angle=30, mu=0.3).is_self_holding

    def test_invalid_cone_angle(self):
        with pytest.raises(ValueError):
            ConeClutch(300, 225, cone_angle=0)
        with pytest.raises(ValueError):
            ConeClutch(300, 225, cone_angle=60)


class TestCentrifugalClutch:
    def make_clutch(self):
        return CentrifugalClutch(drum_radius=100, shoe_mass=0.5, cg_radius=80,
                                 n_shoes=4, spring_force=200, mu=0.3)

    def test_engagement_speed(self):
        clutch = self.make_clutch()
        # w_e = sqrt(S / (m * r_g)), r_g in m
        expected = math.sqrt(200 / (0.5 * 0.080))
        assert clutch.engagement_speed == pytest.approx(expected)
        assert clutch.engagement_speed_rpm == pytest.approx(expected * 60 / (2 * math.pi))

    def test_no_torque_below_engagement(self):
        clutch = self.make_clutch()
        assert clutch.torque(0.5 * clutch.engagement_speed) == 0.0
        assert clutch.shoe_normal_force(0.0) == 0.0

    def test_torque_above_engagement(self):
        clutch = self.make_clutch()
        omega = 2 * clutch.engagement_speed
        normal = 0.5 * omega ** 2 * 0.080 - 200
        assert clutch.torque(omega) == pytest.approx(4 * 0.3 * normal * 100)

    def test_speed_for_torque_round_trip(self):
        clutch = self.make_clutch()
        omega = 2 * clutch.engagement_speed
        torque = clutch.torque(omega)
        assert clutch.speed_for_torque(torque) == pytest.approx(omega)

    def test_contact_pressure(self):
        clutch = self.make_clutch()
        omega = 2 * clutch.engagement_speed
        assert clutch.contact_pressure(omega, 2000.0) == pytest.approx(
            clutch.shoe_normal_force(omega) / 2000.0)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            CentrifugalClutch(100, 0.5, 120, 4)  # cg outside drum
        with pytest.raises(ValueError):
            CentrifugalClutch(100, 0.5, 80, 1)  # too few shoes
        with pytest.raises(ValueError):
            CentrifugalClutch(100, 0.5, 80, 4, spring_force=-1)
        with pytest.raises(ValueError):
            self.make_clutch().torque(-1.0)
