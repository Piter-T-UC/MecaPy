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
    involute,
    inverse_involute,
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


class TestGeometryExtras:
    """Full standard-geometry property set (x = 0 regression)."""

    def test_rack_constants(self):
        """Clearance, working and whole depth from the basic rack."""
        gear = SpurGear(teeth=20, module=2.5)
        assert gear.clearance == pytest.approx(0.625)
        assert gear.working_depth == pytest.approx(5.0)
        assert gear.whole_depth == pytest.approx(5.625)

    def test_thickness_and_pitches(self):
        """Tooth thickness pi*m/2 and base pitch pi*m*cos(alpha)."""
        gear = SpurGear(teeth=20, module=2.5)
        assert gear.tooth_thickness == pytest.approx(3.927, abs=1e-3)
        assert gear.base_pitch == pytest.approx(7.380, abs=1e-3)

    def test_radii(self):
        """Radius accessors are half the diameters."""
        gear = SpurGear(teeth=20, module=2.5)
        assert gear.pitch_radius == pytest.approx(25.0)
        assert gear.base_radius == pytest.approx(23.492, abs=1e-3)
        assert gear.outside_radius == pytest.approx(27.5)
        assert gear.root_radius == pytest.approx(21.875)

    def test_involute_round_trip(self):
        """inv(20 deg) and its inverse."""
        angle = math.radians(20)
        assert involute(angle) == pytest.approx(0.0149044, abs=1e-6)
        assert inverse_involute(involute(angle)) == pytest.approx(
            angle, abs=1e-9)
        assert inverse_involute(0.0) == 0.0
        with pytest.raises(ValueError):
            inverse_involute(-0.01)

    def test_default_shift_zero(self):
        """Standard gears have x = 0 and an unchanged repr."""
        gear = SpurGear(teeth=20, module=2.5)
        assert gear.profile_shift == 0.0
        assert "x=" not in repr(gear)


class TestProfileShift:
    """Profile-shifted (x != 0) geometry."""

    def test_shifted_geometry(self):
        """z=20, m=2.5, x=0.3 addendum/dedendum and diameters."""
        gear = SpurGear(teeth=20, module=2.5, profile_shift=0.3)
        assert gear.addendum == pytest.approx(3.25)
        assert gear.dedendum == pytest.approx(2.375)
        assert gear.outside_diameter == pytest.approx(56.5)
        assert gear.root_diameter == pytest.approx(45.25)
        assert gear.tooth_thickness == pytest.approx(4.473, abs=1e-3)

    def test_shift_invariants(self):
        """Whole depth and clearance do not change with x."""
        gear = SpurGear(teeth=20, module=2.5, profile_shift=0.3)
        assert gear.whole_depth == pytest.approx(5.625)
        assert gear.clearance == pytest.approx(0.625)

    def test_repr_shows_shift(self):
        """repr includes x only when nonzero."""
        gear = SpurGear(teeth=20, module=2.5, profile_shift=0.3)
        assert "x=0.3" in repr(gear)

    def test_validation(self):
        """x must satisfy -1 < x < 1."""
        with pytest.raises(ValueError):
            SpurGear(teeth=20, module=2.5, profile_shift=1.0)
        with pytest.raises(ValueError):
            SpurGear(teeth=20, module=2.5, profile_shift=-1.0)
        with pytest.raises(ValueError):
            SpurGear(teeth=20, module=2.5, profile_shift=1.5)


class TestChangeMethods:
    """change_teeth() and change_profile_shift() design-iteration setters."""

    def test_change_teeth_updates_geometry(self):
        """All derived geometry follows the new tooth count."""
        gear = SpurGear(teeth=17, module=2.5)
        result = gear.change_teeth(20)
        assert result is gear  # chainable
        assert gear.teeth == 20
        assert gear.pitch_diameter == pytest.approx(50.0)
        fresh = SpurGear(teeth=20, module=2.5)
        assert gear.base_diameter == pytest.approx(fresh.base_diameter)
        assert gear.outside_diameter == pytest.approx(fresh.outside_diameter)

    def test_change_teeth_updates_pair_methods(self):
        """Mesh results recompute with the new count."""
        pinion = SpurGear(teeth=17, module=2.0)
        gear = SpurGear(teeth=51, module=2.0)
        before = pinion.center_distance_with(gear)
        pinion.change_teeth(20)
        assert pinion.center_distance_with(gear) == pytest.approx(before + 3.0)

    def test_change_teeth_validation(self):
        """Same rules as the constructor; gear left untouched on error."""
        gear = SpurGear(teeth=17, module=2.5)
        with pytest.raises(ValueError):
            gear.change_teeth(3)  # below cylindrical minimum
        with pytest.raises(ValueError):
            gear.change_teeth(20.5)
        assert gear.teeth == 17

    def test_change_profile_shift(self):
        """Shifted geometry and undercut check follow the new x."""
        gear = SpurGear(teeth=14, module=2.5)
        assert gear.is_undercut
        result = gear.change_profile_shift(0.4)
        assert result is gear  # chainable
        assert not gear.is_undercut
        assert gear.addendum == pytest.approx(2.5 * 1.4)
        fresh = SpurGear(teeth=14, module=2.5, profile_shift=0.4)
        assert gear.outside_diameter == pytest.approx(fresh.outside_diameter)
        assert gear.tooth_thickness == pytest.approx(fresh.tooth_thickness)

    def test_change_profile_shift_validation(self):
        """x must satisfy -1 < x < 1; gear left untouched on error."""
        gear = SpurGear(teeth=20, module=2.5, profile_shift=0.2)
        with pytest.raises(ValueError):
            gear.change_profile_shift(1.0)
        with pytest.raises(ValueError):
            gear.change_profile_shift(-1.5)
        assert gear.profile_shift == pytest.approx(0.2)

    def test_chained_iteration(self):
        """The two setters chain for quick what-if loops."""
        gear = SpurGear(teeth=14, module=2.0)
        od = gear.change_teeth(18).change_profile_shift(0.1).outside_diameter
        assert od == pytest.approx(18 * 2.0 + 2 * 2.0 * 1.1)


class TestUndercut:
    """Undercut check with and without profile shift."""

    def test_small_pinion_undercut(self):
        """12 teeth at 20 deg is undercut; x_min = 1 - z sin^2/2."""
        pinion = SpurGear(teeth=12, module=2.0)
        assert pinion.min_profile_shift == pytest.approx(0.29813, abs=1e-4)
        assert pinion.is_undercut

    def test_shift_cures_undercut(self):
        """x = 0.3 >= x_min avoids the undercut."""
        pinion = SpurGear(teeth=12, module=2.0, profile_shift=0.3)
        assert not pinion.is_undercut

    def test_large_gear_not_undercut(self):
        """20 teeth needs no shift (x_min < 0)."""
        gear = SpurGear(teeth=20, module=2.0)
        assert gear.min_profile_shift == pytest.approx(-0.16978, abs=1e-4)
        assert not gear.is_undercut

    def test_helical_limit(self):
        """Helix relaxes the limit via cos(beta) and phi_t."""
        gear = HelicalGear(teeth=20, module=3.0, helix_angle=25.0,
                           hand="right")
        assert gear.min_profile_shift == pytest.approx(-0.53238, abs=1e-4)


class TestShiftedMesh:
    """Working pressure angle and center distance for shifted pairs."""

    def test_working_values(self):
        """12/24 pair, m=2, pinion x=0.3."""
        pinion = SpurGear(teeth=12, module=2.0, profile_shift=0.3)
        gear = SpurGear(teeth=24, module=2.0)
        assert pinion.working_pressure_angle_with(gear) == pytest.approx(
            22.317, abs=1e-3)
        assert pinion.center_distance_with(gear) == pytest.approx(36.0)
        assert pinion.working_center_distance_with(gear) == pytest.approx(
            36.568, abs=1e-3)

    def test_zero_shift_matches_reference(self):
        """With x1 + x2 = 0 the working values equal the reference."""
        pinion = SpurGear(teeth=20, module=2.5)
        gear = SpurGear(teeth=40, module=2.5)
        assert pinion.working_pressure_angle_with(gear) == pytest.approx(
            20.0, abs=1e-9)
        assert pinion.working_center_distance_with(gear) == pytest.approx(
            75.0, abs=1e-9)

    def test_shifted_contact_ratio(self):
        """Contact ratio uses the working center distance."""
        pinion = SpurGear(teeth=12, module=2.0, profile_shift=0.3)
        gear = SpurGear(teeth=24, module=2.0)
        assert pinion.contact_ratio_with(gear) == pytest.approx(
            1.410, abs=1e-3)


class TestInterference:
    """Tip (involute) interference checks."""

    def test_large_ratio_interferes(self):
        """10/40 at m=2: the gear tip passes the pinion limit."""
        pinion = SpurGear(teeth=10, module=2.0)
        gear = SpurGear(teeth=40, module=2.0)
        assert pinion.has_interference_with(gear)

    def test_standard_pair_clean(self):
        """20/40 at m=2 has no interference."""
        pinion = SpurGear(teeth=20, module=2.0)
        gear = SpurGear(teeth=40, module=2.0)
        assert not pinion.has_interference_with(gear)

    def test_shift_cures_interference(self):
        """12/24 interferes at x=0; pinion x=0.3 cures it."""
        gear = SpurGear(teeth=24, module=2.0)
        assert SpurGear(teeth=12, module=2.0).has_interference_with(gear)
        shifted = SpurGear(teeth=12, module=2.0, profile_shift=0.3)
        assert not shifted.has_interference_with(gear)

    def test_symmetric(self):
        """The check gives the same answer from either member."""
        pinion = SpurGear(teeth=10, module=2.0)
        gear = SpurGear(teeth=40, module=2.0)
        assert (pinion.has_interference_with(gear)
                == gear.has_interference_with(pinion))


class TestDescribe:
    """describe() geometry report."""

    def test_spur_content(self):
        """All standard parameters appear with symbol and unit."""
        report = SpurGear(teeth=20, module=2.5).describe()
        assert isinstance(report, str)
        assert "addendum (ha) = 2.500 mm" in report
        assert "dedendum (hf) = 3.125 mm" in report
        assert "clearance (c) = 0.625 mm" in report
        assert "tooth thickness (s) = 3.927 mm" in report
        assert "profile shift coefficient (x) = 0.000" in report
        assert "min teeth without undercut (Zmin) = 18" in report
        assert "undercut = no" in report
        assert "helix angle" not in report

    def test_helical_content(self):
        """Helical report adds the transverse/helix block."""
        report = HelicalGear(teeth=20, module=3.0, helix_angle=25.0,
                             hand="right").describe()
        assert "helix angle (beta) = 25.000 deg" in report
        assert "hand = right" in report
        assert "transverse module (mt) = 3.310 mm" in report

    def test_herringbone_has_no_hand_line(self):
        """Herringbone (hand=None) omits the hand line."""
        report = HerringboneGear(teeth=30, module=3.0,
                                 helix_angle=30.0).describe()
        assert "helix angle (beta) = 30.000 deg" in report
        assert "hand =" not in report


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


class TestForces:
    """Force and moment export across the gear types."""

    def test_spur_force_report(self):
        """Spur: no thrust, torque = Ft * pitch radius = 9549 P / n."""
        gear = SpurGear(teeth=20, module=2.0)
        f = gear.force_report(power_kw=10.0, speed_rpm=1800.0)
        assert f["Fa"] == 0.0
        assert f["moment"] == 0.0
        assert f["Fr"] == pytest.approx(f["Ft"] * math.tan(math.radians(20)))
        assert f["torque"] == pytest.approx(f["Ft"] * gear.pitch_radius / 1000)
        assert f["torque"] == pytest.approx(9549 * 10.0 / 1800.0, rel=1e-3)

    def test_helical_force_report(self):
        """Helical: axial thrust Ft * tan(beta) and a non-zero moment."""
        gear = HelicalGear(teeth=20, module=2.0, helix_angle=25.0,
                           hand="right")
        f = gear.force_report(power_kw=10.0, speed_rpm=1800.0)
        assert f["Fa"] == pytest.approx(f["Ft"] * math.tan(math.radians(25)))
        assert f["moment"] > 0
        # Radial uses the transverse pressure angle.
        phi_t = math.radians(gear.transverse_pressure_angle)
        assert f["Fr"] == pytest.approx(f["Ft"] * math.tan(phi_t))

    def test_herringbone_no_thrust(self):
        """Herringbone thrust cancels: Fa and moment are zero."""
        gear = HerringboneGear(teeth=30, module=3.0, helix_angle=30.0)
        f = gear.force_report(power_kw=5.0, speed_rpm=1000.0)
        assert f["Fa"] == 0.0
        assert f["moment"] == 0.0
        assert f["Ft"] > 0

    def test_force_report_validation(self):
        """Non-positive power or speed raises."""
        gear = SpurGear(teeth=20, module=2.0)
        with pytest.raises(ValueError):
            gear.force_report(power_kw=0.0, speed_rpm=1800.0)
        with pytest.raises(ValueError):
            gear.force_report(power_kw=10.0, speed_rpm=0.0)

    def test_describe_forces_text(self):
        """The report is a formatted string naming the components."""
        gear = SpurGear(teeth=20, module=2.0, name="pinion")
        text = gear.describe_forces(power_kw=10.0, speed_rpm=1800.0)
        assert "tangential force (Ft)" in text
        assert "torque (T)" in text
        assert "pinion" in text

    def test_bevel_force_report(self):
        """Bevel forces at the mean radius; torque = Ft * r_mean."""
        pinion = BevelGear(teeth=16, module=3.0, face_width=12.0)
        gear = BevelGear(teeth=32, module=3.0)
        f = pinion.force_report(power_kw=5.0, speed_rpm=1200.0, mate=gear)
        r_mean = pinion.mean_radius_with(gear)
        assert f["Ft"] > 0 and f["Fr"] > 0 and f["Fa"] > 0
        assert f["torque"] == pytest.approx(f["Ft"] * r_mean / 1000)
        # Radial and axial split Ft*tan(phi) by the cone angle.
        gamma = math.radians(pinion.pitch_cone_angle_with(gear))
        ft_tan = f["Ft"] * math.tan(math.radians(20))
        assert f["Fr"] == pytest.approx(ft_tan * math.cos(gamma))
        assert f["Fa"] == pytest.approx(ft_tan * math.sin(gamma))

    def test_worm_drive_forces_couple(self):
        """Worm axial equals wheel tangential; separating is shared."""
        worm = Worm(starts=2, module=4.0, pitch_diameter=50.0)
        wheel = WormWheel(teeth=40, module=4.0)
        fw = worm.force_report(power_kw=3.0, speed_rpm=1750.0, wheel=wheel)
        assert fw["Ft"] > 0 and fw["Fr"] > 0 and fw["Fa"] > 0
        # Worm axial thrust is the wheel tangential (output) force.
        assert fw["Fa"] == pytest.approx(fw["wheel_tangential"])
        # Wheel-side view: its tangential is the worm axial, and vice versa.
        wheel_speed = 1750.0 / worm.ratio_with(wheel)
        fg = wheel.force_report(power_kw=3.0, speed_rpm=wheel_speed, worm=worm)
        assert fg["Ft"] == pytest.approx(fw["Fa"])
        assert fg["Fa"] == pytest.approx(fw["Ft"])
        assert fg["Fr"] == pytest.approx(fw["Fr"])

    def test_rack_force_report(self):
        """Rack reacts the pinion tangential force, no torque."""
        rack = Rack(module=2.0)
        pinion = SpurGear(teeth=20, module=2.0)
        f = rack.force_report(pinion, power_kw=1.0, speed_rpm=600.0)
        assert f["torque"] is None
        assert f["Ft"] == pytest.approx(
            pinion.tangential_force(1.0, 600.0))
        assert f["Fr"] == pytest.approx(f["Ft"] * math.tan(math.radians(20)))
