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


def _two_stage_train():
    """A 10 kW / 1200 rpm two-stage spur reducer, 0.98 per stage."""
    p = SpurGear(17, module=3.0, face_width=40.0,
                 power_kw=10.0, speed_rpm=1200.0)
    mid_in = SpurGear(51, module=3.0, face_width=40.0)
    mid_out = SpurGear(18, module=4.0, face_width=50.0)
    out = SpurGear(54, module=4.0, face_width=50.0)
    t = (Transmission(name="reducer")
         .add_stage(p, mid_in, efficiency=0.98)
         .add_stage(mid_out, out, efficiency=0.98))
    return t, p, mid_in, mid_out, out


class TestTransmissionAGMA:
    """AGMA bending/pitting rating of a whole train."""

    def test_one_rating_per_stage(self):
        """rate_agma returns a rating per stage, tagged with its index."""
        t = _two_stage_train()[0]
        ratings = t.rate_agma(Qv=8, hardness_HB=300)
        assert len(ratings) == 2
        assert [r.stage_index for r in ratings] == [0, 1]

    def test_matches_per_mesh_rating(self):
        """Stage 0 agrees with rating the mesh directly."""
        t, p, mid_in, _, _ = _two_stage_train()
        stage0 = t.rate_agma(Qv=8, hardness_HB=300)[0]
        direct = p.rate_agma(mid_in, power_kw=10.0, pinion_speed_rpm=1200.0,
                             Qv=8, hardness_HB=300)
        assert stage0.bending_stress_pinion == pytest.approx(
            direct.bending_stress_pinion)
        assert stage0.contact_stress == pytest.approx(direct.contact_stress)

    def test_second_stage_uses_propagated_operating_point(self):
        """Stage 1 is rated at the reduced speed and derated power."""
        t, _, _, mid_out, _ = _two_stage_train()
        stage1 = t.rate_agma(Qv=8, hardness_HB=300)[1]
        speed = 1200.0 / (51 / 17)          # 400 rpm on the middle shaft
        assert mid_out.speed_rpm == pytest.approx(speed)
        assert stage1.pitch_line_velocity == pytest.approx(
            math.pi * mid_out.pitch_diameter * speed / 60000.0)
        assert stage1.Ft == pytest.approx(
            1000 * 10.0 * 0.98 / stage1.pitch_line_velocity)

    def test_stage_kwargs_override(self):
        """A per-stage Qv changes only that stage's dynamic factor."""
        t = _two_stage_train()[0]
        base = t.rate_agma(Qv=8, hardness_HB=300)
        tuned = t.rate_agma(Qv=8, hardness_HB=300,
                            stage_kwargs=[None, {"Qv": 6}])
        assert tuned[0].Kv == pytest.approx(base[0].Kv)
        assert tuned[1].Kv > base[1].Kv          # coarser quality, higher Kv

    def test_stage_kwargs_wrong_length_raises(self):
        t = _two_stage_train()[0]
        with pytest.raises(ValueError):
            t.rate_agma(hardness_HB=300, stage_kwargs=[{}])

    def test_stage_kwargs_cannot_set_operating_point(self):
        """The train owns power/speed; a stage override must not fight it."""
        t = _two_stage_train()[0]
        with pytest.raises(ValueError, match="operating point"):
            t.rate_agma(hardness_HB=300,
                        stage_kwargs=[{"power_kw": 5.0}, None])

    def test_explicit_rating_inputs_reach_every_stage(self):
        """Named arguments are forwarded, not swallowed."""
        t = _two_stage_train()[0]
        ratings = t.rate_agma(hardness_HB=300, Ko=1.5, Ks=1.1,
                              reliability=0.999, temperature_celsius=130)
        for r in ratings:
            assert r.Ko == pytest.approx(1.5)
            assert r.Ks == pytest.approx(1.1)
            assert r.YZ > 1.0                      # 99.9% is stricter than 99%
            assert r.temperature_factor == pytest.approx((220 + 130) / 330)

    def test_unknown_rating_input_raises(self):
        """A typo'd input is a TypeError, not a silently ignored kwarg."""
        t = _two_stage_train()[0]
        with pytest.raises(TypeError):
            t.rate_agma(hardness_HB=300, Kv=1.2)

    def test_worm_stage_skipped_and_reported(self):
        """A worm stage is left unrated rather than raising."""
        p = SpurGear(17, module=3.0, face_width=40.0,
                     power_kw=5.0, speed_rpm=1000.0)
        g = SpurGear(51, module=3.0, face_width=40.0)
        worm = Worm(starts=2, module=3.0, pitch_diameter=50.0)
        wheel = WormWheel(40, module=3.0, face_width=30.0)
        t = (Transmission()
             .add_stage(p, g)
             .add_stage(worm, wheel))
        assert len(t.rate_agma(hardness_HB=300)) == 1
        unrated = t.agma_unrated_stages
        assert len(unrated) == 1 and unrated[0][0] == 1

    def test_speed_increasing_stage_swaps_pinion(self):
        """A driven gear smaller than its driver still rates."""
        driver = SpurGear(51, module=3.0, face_width=40.0,
                          power_kw=10.0, speed_rpm=400.0)
        driven = SpurGear(17, module=3.0, face_width=40.0)
        t = Transmission().add_stage(driver, driven)
        rating = t.rate_agma(hardness_HB=300)[0]
        assert rating.pinion is driven          # smaller member is the pinion
        assert rating.gear is driver

    def test_governing_picks_the_worst_stage(self):
        """agma_governing reports the minimum SF/SH and their stages."""
        t = _two_stage_train()[0]
        ratings = t.rate_agma(Qv=8, hardness_HB=300)
        g = t.agma_governing(Qv=8, hardness_HB=300)
        expected_sf = min(sf for r in ratings
                          for sf in (r.SF_pinion, r.SF_gear)
                          if sf is not None)
        assert g["SF"] == pytest.approx(expected_sf)
        assert g["SH"] == pytest.approx(min(r.SH for r in ratings))
        assert g["SF_stage"] in (0, 1) and g["SH_stage"] in (0, 1)

    def test_recomputes_after_gear_change(self):
        """Editing a gear then re-rating gives new stresses (no stale cache)."""
        t, p, _, _, _ = _two_stage_train()
        before = t.rate_agma(Qv=8, hardness_HB=300)[0].bending_stress_pinion
        p.change_teeth(24)
        after = t.rate_agma(Qv=8, hardness_HB=300)[0].bending_stress_pinion
        assert after != pytest.approx(before)

    def test_missing_operating_point_raises(self):
        """Rating a train whose input gear has no power/speed raises."""
        p = SpurGear(17, module=3.0, face_width=40.0)
        t = Transmission().add_stage(p, SpurGear(51, module=3.0,
                                                 face_width=40.0))
        with pytest.raises(ValueError):
            t.rate_agma(hardness_HB=300)

    def test_missing_face_width_names_the_stage(self):
        """A stage that cannot be rated reports its index."""
        p = SpurGear(17, module=3.0, face_width=40.0,
                     power_kw=10.0, speed_rpm=1200.0)
        mid_in = SpurGear(51, module=3.0, face_width=40.0)
        mid_out = SpurGear(18, module=4.0)          # no face width
        out = SpurGear(54, module=4.0)
        t = (Transmission()
             .add_stage(p, mid_in)
             .add_stage(mid_out, out))
        with pytest.raises(ValueError, match="Stage 1"):
            t.rate_agma(hardness_HB=300)

    def test_stresses_without_any_material_data(self):
        """Stresses come out with no hardness given — the sizing workflow."""
        t = _two_stage_train()[0]
        ratings = t.rate_agma(Qv=8)
        for r in ratings:
            assert r.has_allowables is False
            assert r.bending_stress_pinion > 0
            assert r.contact_stress > 0

    def test_required_strengths_round_trip(self):
        """A material meeting required_St/_Sc lands on the target SF/SH."""
        t = _two_stage_train()[0]
        r = t.rate_agma(Qv=8)[0]
        st, sc = r.required_St(1.5), r.required_Sc(1.2)
        rated = t.rate_agma(Qv=8, St=st, Sc=sc)[0]
        assert min(rated.SF_pinion, rated.SF_gear) == pytest.approx(1.5)
        assert rated.SH == pytest.approx(1.2)

    def test_idler_gets_ki_factor(self):
        """A gear reused across two stages (an idler) is flagged 1.42 —
        only on its own side of each mesh, not on the other member.
        """
        driver = SpurGear(20, module=2.0, face_width=15.0,
                          power_kw=5.0, speed_rpm=1800.0)
        idler = SpurGear(30, module=2.0, face_width=15.0)
        driven = SpurGear(40, module=2.0, face_width=15.0)
        t = (Transmission()
             .add_stage(driver, idler)
             .add_stage(idler, driven))
        stage0, stage1 = t.rate_agma()
        assert stage0.Ki_pinion == pytest.approx(1.0)   # driver: one mesh
        assert stage0.Ki_gear == pytest.approx(1.42)    # idler as driven
        assert stage1.Ki_pinion == pytest.approx(1.42)  # idler as driver
        assert stage1.Ki_gear == pytest.approx(1.0)      # driven: one mesh

    def test_non_idler_stages_default_ki_to_one(self):
        """A plain two-stage train (no shared gear) gets Ki=1.0 throughout."""
        t = _two_stage_train()[0]
        for r in t.rate_agma(Qv=8, hardness_HB=300):
            assert r.Ki_pinion == pytest.approx(1.0)
            assert r.Ki_gear == pytest.approx(1.0)

    def test_idler_ki_can_be_overridden_per_stage(self):
        """stage_kwargs still wins over the auto-detected idler factor."""
        driver = SpurGear(20, module=2.0, face_width=15.0,
                          power_kw=5.0, speed_rpm=1800.0)
        idler = SpurGear(30, module=2.0, face_width=15.0)
        driven = SpurGear(40, module=2.0, face_width=15.0)
        t = (Transmission()
             .add_stage(driver, idler)
             .add_stage(idler, driven))
        stage0, _ = t.rate_agma(stage_kwargs=[{"Ki_gear": 1.0}, None])
        assert stage0.Ki_gear == pytest.approx(1.0)

    def test_summary_without_material_suggests_strengths(self):
        """The train summary degrades to a material-selection hint."""
        t = _two_stage_train()[0]
        text = t.agma_summary(Qv=8)
        assert "Material selection" in text
        assert "Governing bending" not in text

    def test_summary_covers_every_stage(self):
        t = _two_stage_train()[0]
        text = t.agma_summary(Qv=8, hardness_HB=300)
        assert "Stage 0" in text and "Stage 1" in text
        assert "Governing bending" in text


class TestInternalMeshRules:
    """_check_mesh acceptance and rejection for internal gears."""

    def test_internal_stage_accepted(self):
        """A pinion inside a bigger ring is a valid stage."""
        t = Transmission().add_stage(
            SpurGear(20, module=2.0),
            SpurGear(80, module=2.0, internal=True))
        assert t.overall_ratio == pytest.approx(4.0)

    def test_ring_can_drive(self):
        """A ring gear may be the driver of its stage."""
        t = Transmission().add_stage(
            SpurGear(80, module=2.0, internal=True),
            SpurGear(20, module=2.0))
        assert t.overall_ratio == pytest.approx(0.25)

    def test_two_internal_gears_rejected(self):
        """Two rings have no way to mesh."""
        with pytest.raises(ValueError, match="Two internal gears"):
            Transmission().add_stage(
                SpurGear(80, module=2.0, internal=True),
                SpurGear(60, module=2.0, internal=True))

    def test_ring_must_be_larger(self):
        """The internal member must have more teeth than its pinion."""
        with pytest.raises(ValueError, match="more teeth"):
            Transmission().add_stage(
                SpurGear(60, module=2.0),
                SpurGear(40, module=2.0, internal=True))

    def test_internal_rack_rejected(self):
        """A ring gear cannot drive a rack."""
        with pytest.raises(ValueError, match="external cylindrical"):
            Transmission().add_stage(
                SpurGear(80, module=2.0, internal=True),
                Rack(module=2.0))

    def test_internal_helical_needs_the_same_hand(self):
        """Opposite hands are the external rule; internal is the same hand."""
        ring = HelicalGear(80, module=2.0, helix_angle=20.0, hand="right",
                           internal=True)
        same = HelicalGear(20, module=2.0, helix_angle=20.0, hand="right")
        opposite = HelicalGear(20, module=2.0, helix_angle=20.0, hand="left")
        Transmission().add_stage(same, ring)
        with pytest.raises(ValueError, match="same hand"):
            Transmission().add_stage(opposite, ring)

    def test_external_helical_still_needs_opposite_hands(self):
        """The external rule is unchanged."""
        right = HelicalGear(20, module=2.0, helix_angle=20.0, hand="right")
        other = HelicalGear(60, module=2.0, helix_angle=20.0, hand="right")
        with pytest.raises(ValueError, match="opposite hands"):
            Transmission().add_stage(right, other)

    def test_center_distance_is_a_difference(self):
        """An internal stage subtracts the pitch radii."""
        t = Transmission().add_stage(
            SpurGear(20, module=2.0),
            SpurGear(80, module=2.0, internal=True))
        assert t.center_distance(0) == pytest.approx(60.0)


class TestInternalTrainValue:
    """An internal mesh does not reverse the direction of rotation."""

    def test_single_internal_stage_is_positive(self):
        """Input and output turn the same way."""
        t = Transmission().add_stage(
            SpurGear(20, module=2.0),
            SpurGear(80, module=2.0, internal=True))
        assert t.train_value == pytest.approx(0.25)

    def test_single_external_stage_is_negative(self):
        """The external case is unchanged."""
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     SpurGear(80, module=2.0))
        assert t.train_value == pytest.approx(-0.25)

    def test_mixed_train(self):
        """One external plus one internal mesh reverses once overall."""
        a = SpurGear(17, module=2.0)
        b = SpurGear(51, module=2.0)
        c = SpurGear(18, module=2.5)
        d = SpurGear(72, module=2.5, internal=True)
        t = Transmission().add_stage(a, b).add_stage(c, d)
        assert t.train_value == pytest.approx(-1 / 12)


class TestRotationSenses:
    """Direction of rotation propagated through a train."""

    def test_external_mesh_flips(self):
        """An external mesh reverses the sense."""
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     SpurGear(60, module=2.0))
        assert t.rotation_senses() == [{"driver": 1, "driven": -1}]

    def test_internal_mesh_preserves(self):
        """An internal mesh keeps the sense."""
        t = Transmission().add_stage(
            SpurGear(20, module=2.0),
            SpurGear(80, module=2.0, internal=True))
        assert t.rotation_senses() == [{"driver": 1, "driven": 1}]

    def test_input_sense_is_honoured(self):
        """The input sense flips the whole train, and must be +1 or -1."""
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     SpurGear(60, module=2.0))
        assert t.rotation_senses(-1) == [{"driver": -1, "driven": 1}]
        with pytest.raises(ValueError, match="input_sense"):
            t.rotation_senses(0)

    def test_idler_train_agrees_with_train_value(self):
        """Two external meshes bring the output back to the input sense."""
        first = SpurGear(20, module=2.0)
        idler = SpurGear(30, module=2.0)
        last = SpurGear(60, module=2.0)
        t = Transmission().add_stage(first, idler).add_stage(idler, last)
        senses = t.rotation_senses()
        assert senses[-1]["driven"] == 1
        assert t.train_value > 0

    def test_rack_has_no_rotation(self):
        """A rack translates, so its sense is None."""
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     Rack(module=2.0))
        assert t.rotation_senses() == [{"driver": 1, "driven": None}]

    def test_bevel_train_rejected(self):
        """Intersecting axes have no single sense in the drawing plane."""
        t = Transmission().add_stage(BevelGear(20, module=2.0),
                                     BevelGear(40, module=2.0))
        with pytest.raises(ValueError, match="parallel-axis"):
            t.rotation_senses()


class TestStageLayout:
    """Drawing geometry, computed without matplotlib."""

    def test_centers_advance_by_the_center_distance(self):
        """Gears are laid out along +x, one center distance apart."""
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     SpurGear(60, module=2.0))
        layout = t.stage_layout()
        assert [item["center"] for item in layout] == [(0.0, 0.0), (80.0, 0.0)]

    def test_compound_stages_are_coaxial(self):
        """A stage driver shares its shaft with the previous driven gear."""
        a = SpurGear(17, module=2.0)
        b = SpurGear(51, module=2.0)
        c = SpurGear(18, module=2.5)
        d = SpurGear(72, module=2.5)
        t = Transmission().add_stage(a, b).add_stage(c, d)
        centers = {id(item["element"]): item["center"]
                   for item in t.stage_layout()}
        assert centers[id(b)] == centers[id(c)]
        assert centers[id(d)][0] == pytest.approx(68.0 + 112.5)

    def test_internal_stage_places_the_pinion_inside(self):
        """The centers are only a difference of radii apart."""
        pinion = SpurGear(20, module=2.0)
        ring = SpurGear(80, module=2.0, internal=True)
        layout = Transmission().add_stage(pinion, ring).stage_layout()
        assert layout[1]["center"] == (60.0, 0.0)
        assert layout[1]["center"][0] < ring.outside_radius

    def test_idler_appears_once(self):
        """A reused gear object is one body, not two."""
        first = SpurGear(20, module=2.0)
        idler = SpurGear(30, module=2.0)
        last = SpurGear(60, module=2.0)
        t = Transmission().add_stage(first, idler).add_stage(idler, last)
        layout = t.stage_layout()
        assert len(layout) == 3
        assert [item["element"] for item in layout] == [first, idler, last]

    def test_speeds_and_senses_are_reported(self):
        """Speeds propagate from the given input speed."""
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     SpurGear(60, module=2.0))
        layout = t.stage_layout(input_speed_rpm=1800.0)
        assert layout[0]["speed_rpm"] == pytest.approx(1800.0)
        assert layout[1]["speed_rpm"] == pytest.approx(600.0)
        assert layout[1]["sense"] == -1

    def test_speeds_fall_back_to_the_gears(self):
        """Propagated speeds are used when no input speed is passed."""
        t = Transmission().add_stage(
            SpurGear(20, module=2.0, speed_rpm=1800.0, power_kw=5.0),
            SpurGear(60, module=2.0))
        layout = t.stage_layout()
        assert layout[1]["speed_rpm"] == pytest.approx(600.0)

    def test_rack_sits_under_its_pinion(self):
        """The rack pitch line is tangent to the pinion pitch circle."""
        pinion = SpurGear(20, module=2.0)
        t = Transmission().add_stage(pinion, Rack(module=2.0, length=100))
        layout = t.stage_layout()
        assert layout[1]["center"] == (0.0, -pinion.pitch_radius)


class TestTransmissionPlot:
    """Smoke tests: plot() returns a matplotlib Figure."""

    @pytest.fixture(autouse=True)
    def _matplotlib(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        yield
        matplotlib.pyplot.close("all")

    def test_external_train(self):
        """A two-stage external train draws."""
        from matplotlib.figure import Figure
        t = Transmission(name="two stage").add_stage(
            SpurGear(17, module=2.0, speed_rpm=1500.0, power_kw=5.0),
            SpurGear(51, module=2.0))
        assert isinstance(t.plot(show=False), Figure)

    def test_internal_train(self):
        """A pinion in a ring draws."""
        from matplotlib.figure import Figure
        t = Transmission().add_stage(
            SpurGear(20, module=2.0),
            SpurGear(80, module=2.0, internal=True))
        assert isinstance(t.plot(show=False, input_speed_rpm=1200.0), Figure)

    def test_labels_can_be_switched_off(self):
        """labels=False drops the annotations."""
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     SpurGear(60, module=2.0))
        with_labels = t.plot(show=False, labels=True)
        without = t.plot(show=False, labels=False)
        assert len(with_labels.axes[0].texts) > len(without.axes[0].texts)

    def test_rack_train(self):
        """A rack and pinion draws."""
        from matplotlib.figure import Figure
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     Rack(module=2.0, length=120))
        assert isinstance(t.plot(show=False), Figure)

    def test_bevel_train_rejected(self):
        """Intersecting axes cannot be drawn in a plan view."""
        t = Transmission().add_stage(BevelGear(20, module=2.0),
                                     BevelGear(40, module=2.0))
        with pytest.raises(ValueError, match="parallel-axis"):
            t.plot(show=False)

    def test_reuses_a_given_axes(self):
        """Passing ax= draws into that figure."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        t = Transmission().add_stage(SpurGear(20, module=2.0),
                                     SpurGear(60, module=2.0))
        assert t.plot(show=False, ax=ax) is fig
