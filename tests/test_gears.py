"""Tests for gear module."""

import math

import pytest
from mecapy.gears import (
    Gear,
    SpurGear,
    HelicalGear,
    HerringboneGear,
    Rack,
    BevelGear,
    Worm,
    WormWheel,
    PlanetaryGearSet,
)


class TestGear:
    """Test cases for Gear class."""

    def test_gear_creation(self):
        """Test creating a gear object."""
        gear = Gear(teeth=20, module=2.5, material="steel")
        assert gear.teeth == 20
        assert gear.module == 2.5
        assert gear.material == "steel"

    def test_gear_repr(self):
        """Test gear string representation."""
        gear = Gear(teeth=30, module=3.0, material="cast_iron")
        assert "Gear" in repr(gear)
        assert "30" in repr(gear)


class TestSpurGear:
    """Test cases for SpurGear geometry and validation."""

    def test_geometry(self):
        """Standard full-depth geometry for m=2.5, Z=20."""
        gear = SpurGear(teeth=20, module=2.5)
        assert gear.pitch_diameter == pytest.approx(50.0)
        assert gear.base_diameter == pytest.approx(46.985, abs=1e-3)
        assert gear.outside_diameter == pytest.approx(55.0)
        assert gear.root_diameter == pytest.approx(43.75)
        assert gear.circular_pitch == pytest.approx(7.854, abs=1e-3)
        assert gear.addendum == pytest.approx(2.5)
        assert gear.dedendum == pytest.approx(3.125)

    def test_min_teeth_no_undercut(self):
        """Interference limit ceil(2/sin^2(20deg)) = 18."""
        gear = SpurGear(teeth=20, module=2.0)
        assert gear.min_teeth_no_undercut == 18

    def test_contact_ratio(self):
        """Transverse contact ratio for a 20/40 pair, m=2.5."""
        pinion = SpurGear(teeth=20, module=2.5)
        gear = SpurGear(teeth=40, module=2.5)
        assert pinion.contact_ratio_with(gear) == pytest.approx(1.63, abs=0.02)

    def test_center_distance(self):
        """Center distance is (d1 + d2) / 2."""
        pinion = SpurGear(teeth=20, module=2.5)
        gear = SpurGear(teeth=40, module=2.5)
        assert pinion.center_distance_with(gear) == pytest.approx(75.0)

    def test_validation_errors(self):
        """Non-physical inputs raise ValueError."""
        with pytest.raises(ValueError):
            SpurGear(teeth=0, module=2.5)
        with pytest.raises(ValueError):
            SpurGear(teeth=20, module=-1)
        with pytest.raises(ValueError):
            SpurGear(teeth=20, module=2.5, diametral_pitch=10)
        with pytest.raises(ValueError):
            SpurGear(teeth=20)
        with pytest.raises(ValueError):
            SpurGear(teeth=20, module=2.5, pressure_angle=50)


class TestUnits:
    """US-customary diametral pitch input."""

    def test_diametral_pitch_input(self):
        """Pd = 10 teeth/in stores module 2.54 mm."""
        gear = SpurGear(teeth=20, diametral_pitch=10)
        assert gear.module == pytest.approx(2.54)
        assert gear.diametral_pitch == pytest.approx(10.0)

    def test_metric_round_trip(self):
        """diametral_pitch property is 25.4 / module."""
        gear = SpurGear(teeth=20, module=2.5)
        assert gear.diametral_pitch == pytest.approx(10.16)


class TestHelicalGear:
    """Test cases for HelicalGear transverse geometry."""

    def test_transverse_geometry(self):
        """mn=3, beta=25 deg: mt=3.310, phi_t=21.88 deg."""
        gear = HelicalGear(teeth=20, module=3.0, helix_angle=25.0,
                           hand="right")
        assert gear.transverse_module == pytest.approx(3.310, abs=1e-3)
        assert gear.transverse_pressure_angle == pytest.approx(21.88,
                                                               abs=0.01)
        assert gear.pitch_diameter == pytest.approx(20 * 3.310, abs=0.02)

    def test_hand_required(self):
        """Helical gears require helix angle and a valid hand."""
        with pytest.raises(ValueError):
            HelicalGear(teeth=20, module=3.0, helix_angle=25.0)
        with pytest.raises(ValueError):
            HelicalGear(teeth=20, module=3.0, helix_angle=25.0, hand="up")
        with pytest.raises(ValueError):
            HelicalGear(teeth=20, module=3.0)

    def test_herringbone_has_no_hand(self):
        """Herringbone thrust cancels; no net hand."""
        gear = HerringboneGear(teeth=30, module=3.0, helix_angle=30.0)
        assert gear.hand is None
        assert gear.net_axial_force == 0.0
        assert gear.axial_force(1000.0) == 0.0


class TestBevelGear:
    """Test cases for BevelGear cone geometry."""

    def test_pitch_cone_angle(self):
        """16/32 pair at 90 deg: gamma_pinion = atan(0.5)."""
        pinion = BevelGear(teeth=16, module=3.0)
        gear = BevelGear(teeth=32, module=3.0)
        assert pinion.pitch_cone_angle_with(gear) == pytest.approx(
            math.degrees(math.atan(0.5)), abs=1e-6)
        assert gear.pitch_cone_angle_with(pinion) == pytest.approx(
            math.degrees(math.atan(2.0)), abs=1e-6)

    def test_virtual_teeth(self):
        """Tredgold virtual teeth Zv = Z / cos(gamma)."""
        pinion = BevelGear(teeth=16, module=3.0)
        gear = BevelGear(teeth=32, module=3.0)
        gamma = math.atan(0.5)
        assert pinion.virtual_teeth_with(gear) == pytest.approx(
            16 / math.cos(gamma), abs=1e-6)

    def test_lewis_bending_stress(self):
        """Simplified Lewis check returns a positive stress in MPa."""
        pinion = BevelGear(teeth=16, module=3.0, face_width=12.0)
        gear = BevelGear(teeth=32, module=3.0)
        stress = pinion.lewis_bending_stress(500.0, gear)
        assert stress > 0


class TestWorm:
    """Test cases for worm drives."""

    def test_lead_and_lead_angle(self):
        """2-start, m=4, d=50: lead=25.13 mm, lead angle=9.09 deg."""
        worm = Worm(starts=2, module=4.0, pitch_diameter=50.0)
        assert worm.lead == pytest.approx(25.13, abs=0.01)
        assert worm.lead_angle == pytest.approx(9.09, abs=0.01)

    def test_ratio(self):
        """Ratio = wheel teeth / worm starts."""
        worm = Worm(starts=2, module=4.0, pitch_diameter=50.0)
        wheel = WormWheel(teeth=40, module=4.0)
        assert worm.ratio_with(wheel) == pytest.approx(20.0)

    def test_efficiency_decreases_with_friction(self):
        """More friction, less efficiency."""
        worm = Worm(starts=2, module=4.0, pitch_diameter=50.0)
        assert worm.efficiency(0.02) > worm.efficiency(0.08)
        assert 0 < worm.efficiency() < 1

    def test_self_locking(self):
        """Single-start (small lead angle) locks at f=0.1."""
        worm1 = Worm(starts=1, module=2.0, pitch_diameter=50.0)
        assert worm1.is_self_locking(friction_coefficient=0.1)
        worm4 = Worm(starts=4, module=6.0, pitch_diameter=40.0)
        assert not worm4.is_self_locking(friction_coefficient=0.02)

    def test_wheel_default_bronze(self):
        """Worm wheels default to bronze."""
        wheel = WormWheel(teeth=40, module=4.0)
        assert wheel.material == "bronze"


class TestRack:
    """Test cases for Rack."""

    def test_pitch_and_teeth(self):
        """Linear pitch pi*m; teeth from length."""
        rack = Rack(module=2.0, length=200.0)
        assert rack.circular_pitch == pytest.approx(math.pi * 2.0)
        assert rack.n_teeth == pytest.approx(200.0 / (math.pi * 2.0))

    def test_travel_per_pinion_rev(self):
        """Travel per pinion revolution is pi * d_pinion."""
        rack = Rack(module=2.0)
        pinion = SpurGear(teeth=20, module=2.0)
        assert rack.travel_per_pinion_rev(pinion) == pytest.approx(
            math.pi * 40.0)

    def test_linear_velocity(self):
        """v = pi * d * n / 60000 in m/s."""
        rack = Rack(module=2.0)
        pinion = SpurGear(teeth=20, module=2.0)
        assert rack.linear_velocity(pinion, 600.0) == pytest.approx(
            math.pi * 40.0 * 600.0 / 60000.0)


class TestPlanetary:
    """Test cases for PlanetaryGearSet."""

    def test_valid_assembly(self):
        """Zs=24, Zp=18, Zr=60 with 3 planets assembles."""
        ps = PlanetaryGearSet(24, 18, 60, n_planets=3)
        assert ps.sun_teeth == 24
        assert ps.ring_teeth == 60

    def test_ratios(self):
        """Ring fixed sun->carrier = 1 + Zr/Zs; carrier fixed = -Zr/Zs."""
        ps = PlanetaryGearSet(24, 18, 60, n_planets=3)
        assert ps.ratio("sun", "carrier", "ring") == pytest.approx(3.5)
        assert ps.ratio("sun", "ring", "carrier") == pytest.approx(-2.5)
        assert ps.ratio("carrier", "sun", "ring") == pytest.approx(1 / 3.5)

    def test_speeds(self):
        """Speed solution honours the Willis equation."""
        ps = PlanetaryGearSet(24, 18, 60, n_planets=3)
        speeds = ps.speeds("sun", 3500.0, "ring")
        assert speeds["carrier"] == pytest.approx(1000.0)
        assert speeds["ring"] == 0.0

    def test_torque_balance(self):
        """Member torques sum to zero."""
        ps = PlanetaryGearSet(24, 18, 60, n_planets=3)
        torques = ps.torques("sun", 100.0, "ring")
        assert sum(torques.values()) == pytest.approx(0.0)
        assert torques["sun"] == pytest.approx(100.0)

    def test_geometric_condition(self):
        """Ring teeth must equal sun + 2 * planet."""
        with pytest.raises(ValueError):
            PlanetaryGearSet(24, 18, 61, n_planets=3)

    def test_assembly_condition(self):
        """(Zs + Zr) must divide by the planet count."""
        with pytest.raises(ValueError):
            PlanetaryGearSet(24, 18, 60, n_planets=5)

    def test_gear_objects(self):
        """Gear objects work and must share a module."""
        sun = SpurGear(24, module=2.0)
        planet = SpurGear(18, module=2.0)
        ring = SpurGear(60, module=2.0)
        ps = PlanetaryGearSet(sun, planet, ring, n_planets=3)
        assert ps.ratio("sun", "carrier", "ring") == pytest.approx(3.5)
        with pytest.raises(ValueError):
            PlanetaryGearSet(sun, SpurGear(18, module=3.0), ring)
