"""Tests for the belt subsystem (flat belts, V-belts)."""

import math

import pytest

from mecapy import MechaElement
from mecapy.belts import FlatBelt, VBelt, V_BELT_EFFECTIVE_MU


def make_flat_belt(**overrides):
    params = dict(width=50.0, thickness=5.0, driver_diameter=100.0,
                  driven_diameter=300.0, center_distance=800.0,
                  belt_material="polyamide")
    params.update(overrides)
    return FlatBelt(**params)


class TestFlatBeltConstruction:
    def test_is_mecha_element(self):
        assert isinstance(make_flat_belt(), MechaElement)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            make_flat_belt(width=0)
        with pytest.raises(ValueError):
            make_flat_belt(driver_diameter=290, driven_diameter=300,
                           center_distance=1, drive="open")
        with pytest.raises(ValueError):
            make_flat_belt(driver_diameter=100, driven_diameter=300,
                           center_distance=1, drive="crossed")
        with pytest.raises(ValueError):
            make_flat_belt(drive="backwards")
        with pytest.raises(ValueError):
            make_flat_belt(mu=2.0, belt_material=None)
        with pytest.raises(ValueError):
            make_flat_belt(belt_material="unobtainium")


class TestFlatBeltGeometry:
    def test_wrap_angles_sum_to_360_open(self):
        belt = make_flat_belt()
        assert belt.wrap_angle_driver + belt.wrap_angle_driven == pytest.approx(360.0)

    def test_equal_pulleys_180_and_length(self):
        belt = make_flat_belt(driver_diameter=150.0, driven_diameter=150.0)
        assert belt.wrap_angle_driver == pytest.approx(180.0)
        assert belt.wrap_angle_driven == pytest.approx(180.0)
        assert belt.belt_length == pytest.approx(2 * 800.0 + math.pi * 150.0)

    def test_crossed_angles_equal(self):
        belt = make_flat_belt(drive="crossed")
        assert belt.wrap_angle_driver == pytest.approx(belt.wrap_angle_driven)

    def test_crossed_length_exceeds_open(self):
        open_belt = make_flat_belt()
        crossed_belt = make_flat_belt(drive="crossed")
        assert crossed_belt.belt_length > open_belt.belt_length


class TestFlatBeltTensions:
    def test_power_round_trip(self):
        belt = make_flat_belt()
        f1, _ = belt.tensions_for_power(500.0, 1000)
        assert belt.power(f1, 1000) == pytest.approx(500.0, rel=1e-6)

    def test_tension_ratio_matches_capstan_equation(self):
        belt = make_flat_belt()
        f1, f2 = belt.tensions_for_power(500.0, 1000)
        fc = belt.centrifugal_tension(1000)
        assert (f1 - fc) / (f2 - fc) == pytest.approx(belt.tension_ratio)

    def test_tension_difference_equals_power_over_speed(self):
        belt = make_flat_belt()
        f1, f2 = belt.tensions_for_power(500.0, 1000)
        v = belt.belt_speed(1000)
        assert f1 - f2 == pytest.approx(500.0 / v)

    def test_initial_tension_formula(self):
        belt = make_flat_belt()
        f1, f2 = belt.tensions_for_power(500.0, 1000)
        fc = belt.centrifugal_tension(1000)
        assert belt.initial_tension(500.0, 1000) == pytest.approx((f1 + f2) / 2 - fc)

    def test_raises_for_f1_below_fc(self):
        belt = make_flat_belt()
        fc = belt.centrifugal_tension(1000)
        with pytest.raises(ValueError):
            belt.power(fc * 0.5, 1000)

    def test_raises_for_nonpositive_speed_or_power(self):
        belt = make_flat_belt()
        with pytest.raises(ValueError):
            belt.tensions_for_power(500.0, 0)
        with pytest.raises(ValueError):
            belt.tensions_for_power(0.0, 1000)

    def test_raises_without_density_or_allowable(self):
        belt = make_flat_belt(belt_material=None, mu=0.3)
        with pytest.raises(ValueError):
            belt.centrifugal_tension(1000)
        with pytest.raises(ValueError):
            belt.allowable_tension


class TestShigleyFlatBeltDriveEx17_1:
    """Anchor: Shigley Prob. 17-6, which re-derives Ex. 17-1's A-3 polyamide
    drive (d=6in, D=18in, C=96in, Hnom=15hp, Ks=1.25, nd=1.1, n=1750 rpm)
    using Eqs. 17-1, 17-2, 17-9/17-10 directly -- the same equations this
    class implements. Numbers transcribed from the Ch. 17 solutions manual."""

    def make(self):
        return FlatBelt(width=104.9, thickness=3.302, driver_diameter=152.4,
                        driven_diameter=457.2, center_distance=2438.4,
                        belt_material="polyamide")

    def test_wrap_angle(self):
        assert self.make().governing_wrap_angle == pytest.approx(172.86, rel=1e-2)

    def test_belt_length(self):
        # Shigley: L = 230.074 in = 5843.9 mm
        assert self.make().belt_length == pytest.approx(5843.9, rel=1e-3)

    def test_centrifugal_tension(self):
        belt = self.make()
        fc = belt.centrifugal_tension(1750)
        assert fc / 4.44822 == pytest.approx(17.7, rel=2e-2)  # Shigley: Fc = 17.7 lbf

    def test_tensions_for_power(self):
        belt = self.make()
        power_w = 15380.0  # Hd = 15(1.25)(1.1) = 20.625 hp
        f1, f2 = belt.tensions_for_power(power_w, 1750)
        # Shigley's allowable-tension design gives F1 = 289.1 lbf, F2 = 41.5
        # lbf; friction is nearly fully developed in this example, so the
        # incipient-slip minimum tensions computed here land close by.
        assert f1 / 4.44822 == pytest.approx(289.1, rel=2e-2)
        assert f2 / 4.44822 == pytest.approx(41.5, rel=5e-2)


class TestShigleyFlatBeltDriveProb17_3:
    """Anchor: Shigley Prob. 17-3 (A-2 polyamide, double the dimensions of
    Ex. 17-2): b=12in, d=4in, D=8in, C=216in, n=1750 rpm, Hnom=2hp, Ks=1.25.
    Only the friction-independent quantities (Fc, wrap angles, length,
    delta-F) are anchored -- the book's design has underdeveloped friction
    (f' = 0.03 vs f = 0.8 available), so its F1/F2 come from an allowable-
    tension check this simplified class does not implement."""

    def make(self):
        density = 0.037 * 27679.9
        allowable = (60.0 / 0.11) * 0.00689476
        return FlatBelt(width=304.8, thickness=2.794, driver_diameter=101.6,
                        driven_diameter=203.2, center_distance=5486.4,
                        mu=0.8, density=density, allowable_stress=allowable)

    def test_wrap_angles(self):
        belt = self.make()
        assert math.radians(belt.wrap_angle_driver) == pytest.approx(3.123, rel=1e-2)
        assert math.radians(belt.wrap_angle_driven) == pytest.approx(3.160, rel=1e-2)

    def test_belt_length(self):
        assert self.make().belt_length / 25.4 == pytest.approx(450.9, rel=1e-2)

    def test_centrifugal_tension(self):
        fc = self.make().centrifugal_tension(1750)
        assert fc / 4.44822 == pytest.approx(17.0, rel=2e-2)

    def test_delta_f_matches_power_over_speed(self):
        belt = self.make()
        power_w = 2 * 745.7 * 1.25  # Hnom * Ks, nd=1
        v = belt.belt_speed(1750)
        delta_f = power_w / v
        assert delta_f / 4.44822 == pytest.approx(45.0, rel=2e-2)


class TestVBeltConstruction:
    def make_vbelt(self, **overrides):
        params = dict(section="B", driver_diameter=150.0, driven_diameter=300.0,
                      center_distance=600.0)
        params.update(overrides)
        return VBelt(**params)

    def test_is_flat_belt_subclass(self):
        assert isinstance(self.make_vbelt(), FlatBelt)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            self.make_vbelt(section="Z")
        with pytest.raises(ValueError):
            self.make_vbelt(driver_diameter=50.0)  # below B section minimum
        with pytest.raises(ValueError):
            self.make_vbelt(n_belts=0)
        with pytest.raises(ValueError):
            self.make_vbelt(groove_angle=10.0)


class TestVBeltGeometry:
    def test_pitch_length_center_distance_round_trip(self):
        belt = VBelt(section="B", driver_diameter=150.0, driven_diameter=300.0,
                     center_distance=600.0)
        lp = belt.pitch_length
        assert belt.center_distance_for_pitch_length(lp) == pytest.approx(600.0, rel=1e-6)

    def test_designation(self):
        belt = VBelt(section="B", driver_diameter=157.48, driven_diameter=304.8,
                     center_distance=799.47)
        assert belt.designation == "B90"


class TestVBeltWedging:
    def test_default_effective_friction(self):
        belt = VBelt(section="B", driver_diameter=150.0, driven_diameter=300.0,
                     center_distance=600.0)
        assert belt.effective_friction == pytest.approx(V_BELT_EFFECTIVE_MU)
        assert V_BELT_EFFECTIVE_MU == pytest.approx(0.5123)

    def test_vbelt_tension_ratio_exceeds_flat_belt(self):
        vbelt = VBelt(section="B", driver_diameter=150.0, driven_diameter=300.0,
                      center_distance=600.0, mu=0.3)
        flat = FlatBelt(width=16.764, thickness=10.312, driver_diameter=150.0,
                        driven_diameter=300.0, center_distance=600.0, mu=0.3)
        assert vbelt.tension_ratio > flat.tension_ratio


class TestShigleyVBeltDriveProb17_18:
    """Anchor: Shigley Prob. 17-18, two B85 V-belts, d=5.4in, D=16in,
    n=1200 rpm. Geometry/speed/centrifugal-tension quantities are verified
    against the solutions manual; the full rated-power chain (K1/K2/Htab
    table lookups) is out of scope for this simplified class."""

    def make(self):
        return VBelt(section="B", driver_diameter=137.16, driven_diameter=406.4,
                     center_distance=661.67)

    def test_pitch_length(self):
        assert self.make().pitch_length / 25.4 == pytest.approx(86.8, rel=1e-2)

    def test_wrap_angle(self):
        assert self.make().wrap_angle_driver == pytest.approx(156.5, rel=1e-2)

    def test_belt_speed(self):
        v = self.make().belt_speed(1200)
        assert v / 0.00508 == pytest.approx(1696, rel=1e-2)  # ft/min

    def test_centrifugal_tension_matches_kc_0_965(self):
        # Shigley Table 17-16: Kc = 0.965 for section B;
        # Fc[lbf] = Kc*(V[ft/min]/1000)^2
        belt = self.make()
        fc = belt.centrifugal_tension(1200)
        v_fpm = belt.belt_speed(1200) / 0.00508
        expected_lbf = 0.965 * (v_fpm / 1000) ** 2
        assert fc / 4.44822 == pytest.approx(expected_lbf, rel=1e-2)
