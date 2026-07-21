"""Tests for the Transmission class and mesh compatibility."""

import math

import pytest
from mecapy.gears import (
    SpurGear,
    HelicalGear,
    HerringboneGear,
    Rack,
    BevelGear,
    Worm,
    WormWheel,
    Transmission,
)


class TestTransmissionKinematics:
    """Ratios, speeds and torques of simple and compound trains."""

    def test_single_pair(self):
        """20 -> 40 doubles torque and halves speed."""
        t = Transmission().add_stage(SpurGear(20, module=2.5),
                                     SpurGear(40, module=2.5))
        assert t.overall_ratio == pytest.approx(2.0)
        assert t.output_speed(1000.0) == pytest.approx(500.0)
        assert t.output_torque(10.0) == pytest.approx(20.0)
        assert t.center_distance(0) == pytest.approx(75.0)

    def test_pitch_line_velocity(self):
        """First-stage pitch-line velocity at the driver."""
        pinion = SpurGear(20, module=2.5)
        t = Transmission().add_stage(pinion, SpurGear(40, module=2.5))
        expected = math.pi * pinion.pitch_diameter * 1000.0 / 60000.0
        assert t.pitch_line_velocity(0, 1000.0) == pytest.approx(expected)

    def test_compound_train(self):
        """17->68 then 18->54 gives an overall ratio of 12."""
        t = (Transmission()
             .add_stage(SpurGear(17, module=2.0), SpurGear(68, module=2.0))
             .add_stage(SpurGear(18, module=2.0), SpurGear(54, module=2.0)))
        assert t.overall_ratio == pytest.approx(12.0)
        assert t.output_speed(1200.0) == pytest.approx(100.0)

    def test_idler_train_sign(self):
        """An idler leaves the magnitude and flips the direction twice."""
        g1 = SpurGear(20, module=2.0)
        idler = SpurGear(25, module=2.0)
        g3 = SpurGear(40, module=2.0)
        t = Transmission().add_stage(g1, idler).add_stage(idler, g3)
        assert t.overall_ratio == pytest.approx(2.0)
        assert t.train_value == pytest.approx(0.5)  # two sign flips

    def test_worm_stage_ratio(self):
        """Worm stage ratio is wheel teeth / starts."""
        t = Transmission().add_stage(Worm(2, module=4.0, pitch_diameter=50.0),
                                     WormWheel(40, module=4.0))
        assert t.overall_ratio == pytest.approx(20.0)
        assert t.train_value is None

    def test_rack_output(self):
        """A final rack converts speed to linear velocity."""
        pinion = SpurGear(20, module=2.0)
        t = Transmission().add_stage(pinion, Rack(module=2.0))
        expected = math.pi * 40.0 * 600.0 / 60000.0
        assert t.output_speed(600.0) == pytest.approx(expected)
        with pytest.raises(ValueError):
            t.output_torque(10.0)

    def test_empty_transmission(self):
        """Ratio of an empty train raises."""
        with pytest.raises(ValueError):
            Transmission().overall_ratio


class TestMeshCompatibility:
    """ValueError cases from _check_mesh."""

    def test_module_mismatch(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(SpurGear(20, module=2.0),
                                     SpurGear(40, module=2.5))

    def test_pressure_angle_mismatch(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(
                SpurGear(20, module=2.0, pressure_angle=20.0),
                SpurGear(40, module=2.0, pressure_angle=25.0))

    def test_spur_with_helical(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(
                SpurGear(20, module=2.0),
                HelicalGear(40, module=2.0, helix_angle=15.0, hand="left"))

    def test_helical_same_hand(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(
                HelicalGear(20, module=2.0, helix_angle=15.0, hand="right"),
                HelicalGear(40, module=2.0, helix_angle=15.0, hand="right"))

    def test_helical_opposite_hands_ok(self):
        t = Transmission().add_stage(
            HelicalGear(20, module=2.0, helix_angle=15.0, hand="right"),
            HelicalGear(40, module=2.0, helix_angle=15.0, hand="left"))
        assert t.overall_ratio == pytest.approx(2.0)

    def test_helical_angle_mismatch(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(
                HelicalGear(20, module=2.0, helix_angle=15.0, hand="right"),
                HelicalGear(40, module=2.0, helix_angle=20.0, hand="left"))

    def test_herringbone_pair(self):
        t = Transmission().add_stage(
            HerringboneGear(20, module=2.0, helix_angle=30.0),
            HerringboneGear(40, module=2.0, helix_angle=30.0))
        assert t.overall_ratio == pytest.approx(2.0)

    def test_herringbone_with_spur(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(
                HerringboneGear(20, module=2.0, helix_angle=30.0),
                SpurGear(40, module=2.0))

    def test_spur_with_bevel(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(SpurGear(20, module=2.0),
                                     BevelGear(40, module=2.0))

    def test_bevel_pair_ok(self):
        t = Transmission().add_stage(BevelGear(16, module=3.0),
                                     BevelGear(32, module=3.0))
        assert t.overall_ratio == pytest.approx(2.0)

    def test_rack_cannot_drive(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(Rack(module=2.0),
                                     SpurGear(20, module=2.0))

    def test_rack_must_be_last(self):
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     Rack(module=2.0))
        with pytest.raises(ValueError):
            t.add_stage(SpurGear(20, module=2.0), SpurGear(40, module=2.0))

    def test_worm_cannot_be_driven(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(SpurGear(20, module=4.0),
                                     Worm(2, module=4.0, pitch_diameter=50.0))

    def test_worm_needs_worm_wheel(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(Worm(2, module=4.0, pitch_diameter=50.0),
                                     SpurGear(40, module=4.0))

    def test_worm_wheel_needs_worm(self):
        with pytest.raises(ValueError):
            Transmission().add_stage(SpurGear(20, module=4.0),
                                     WormWheel(40, module=4.0))


class TestOperatingPointPropagation:
    """The first gear's power/speed flows onto every downstream gear."""

    def test_auto_propagation_on_add_stage(self):
        """add_stage assigns speed and power to the driven gear."""
        p = SpurGear(20, module=2.0, power_kw=10.0, speed_rpm=1500.0)
        g = SpurGear(60, module=2.0)
        Transmission().add_stage(p, g)
        assert g.speed_rpm == pytest.approx(500.0)   # 1500 / (60/20)
        assert g.power_kw == pytest.approx(10.0)      # lossless

    def test_efficiency_reduces_downstream_power(self):
        """A per-stage efficiency below 1 lowers the driven power."""
        p = SpurGear(20, module=2.0, power_kw=10.0, speed_rpm=1500.0)
        g = SpurGear(60, module=2.0)
        Transmission().add_stage(p, g, efficiency=0.98)
        assert g.power_kw == pytest.approx(9.8)
        assert g.speed_rpm == pytest.approx(500.0)

    def test_efficiency_out_of_range_raises(self):
        p = SpurGear(20, module=2.0, power_kw=10.0, speed_rpm=1500.0)
        with pytest.raises(ValueError):
            Transmission().add_stage(p, SpurGear(60, module=2.0),
                                     efficiency=1.5)

    def test_two_stage_train(self):
        """Power and speed propagate across a compound train.

        The stage-1 driven gear (68 t) and the stage-2 driver (18 t) sit
        on the same shaft, so they share a speed.
        """
        p = SpurGear(17, module=2.0, power_kw=10.0, speed_rpm=1200.0)
        mid_in = SpurGear(68, module=2.0)
        mid_out = SpurGear(18, module=2.0)
        out = SpurGear(54, module=2.0)
        t = (Transmission()
             .add_stage(p, mid_in, efficiency=0.98)
             .add_stage(mid_out, out, efficiency=0.97))
        assert mid_in.speed_rpm == pytest.approx(300.0)   # 1200 / 4
        assert out.speed_rpm == pytest.approx(100.0)      # 300 / 3
        assert out.power_kw == pytest.approx(10.0 * 0.98 * 0.97)
        assert t.output_power == pytest.approx(10.0 * 0.98 * 0.97)

    def test_propagate_requires_first_operating_point(self):
        """propagate() raises when the first gear has no power/speed."""
        p = SpurGear(20, module=2.0)
        t = Transmission().add_stage(p, SpurGear(60, module=2.0))
        with pytest.raises(ValueError):
            t.propagate()

    def test_partial_build_does_not_raise(self):
        """A gear with no operating point builds without auto-propagating."""
        p = SpurGear(20, module=2.0)
        g = SpurGear(60, module=2.0)
        Transmission().add_stage(p, g)   # no raise
        assert g.speed_rpm is None

    def test_repropagate_after_change_teeth(self):
        """Re-running propagate() after editing a gear recomputes speeds."""
        p = SpurGear(20, module=2.0, power_kw=10.0, speed_rpm=1500.0)
        g = SpurGear(60, module=2.0)
        t = Transmission().add_stage(p, g)
        assert g.speed_rpm == pytest.approx(500.0)
        p.change_teeth(24)
        t.propagate()
        assert g.speed_rpm == pytest.approx(600.0)   # 1500 / (60/24)

    def test_output_power_needs_stages(self):
        with pytest.raises(ValueError):
            _ = Transmission().output_power
