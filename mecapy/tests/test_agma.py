"""Tests for the AGMA rating module.

The end-to-end reference is the classic Shigley spur-gear example
(17-tooth pinion, 52-tooth gear, Pd = 10 teeth/in, 20 deg full depth,
F = 1.5 in, 1800 rpm, 4 hp, Qv = 6): sigma_F about 6.15 kpsi
(42.4 MPa) and sigma_H about 68.9 kpsi (475 MPa). The J factors are
digitized from figures, so stresses are checked to 5% and computed
factors to 1%.
"""

import pytest
from mecapy.gears import (
    SpurGear, HelicalGear, Rack, AGMARating, GearMaterial,
)
from mecapy.gears import agma, agma_data
from mecapy.materials import Material


class TestFactors:
    """Unit tests for the individual AGMA factor functions."""

    def test_dynamic_factor(self):
        """Kv for Qv=6 at 4.07 m/s (Shigley example: about 1.377)."""
        assert agma.dynamic_factor(4.069, 6) == pytest.approx(1.38, abs=0.01)
        assert agma.dynamic_factor(0.0, 6) == pytest.approx(1.0)

    def test_dynamic_factor_validity(self):
        """Velocity beyond the Qv limit raises."""
        with pytest.raises(ValueError):
            agma.dynamic_factor(100.0, 5)
        with pytest.raises(ValueError):
            agma.dynamic_factor(5.0, 12)

    def test_load_distribution_factor(self):
        """KH for F=38.1 mm, d=43.18 mm commercial (about 1.22)."""
        kh = agma.load_distribution_factor(38.1, 43.18)
        assert kh == pytest.approx(1.22, abs=0.01)

    def test_rim_thickness_factor(self):
        """KB is 1 for solid gears and grows below mB = 1.2."""
        assert agma.rim_thickness_factor(2.0) == 1.0
        assert agma.rim_thickness_factor(0.8) == pytest.approx(1.648,
                                                               abs=0.01)

    def test_size_factor_bands(self):
        """Ks steps up with tooth size; listed modules are upper bounds."""
        assert agma.size_factor(1.5) == 1.00
        assert agma.size_factor(5.0) == 1.00
        assert agma.size_factor(5.5) == 1.05
        assert agma.size_factor(6.0) == 1.05
        assert agma.size_factor(8.0) == 1.15
        assert agma.size_factor(12.0) == 1.25
        assert agma.size_factor(20.0) == 1.40
        assert agma.size_factor(30.0) == 1.40      # clamped above the table

    def test_size_factor_validation(self):
        with pytest.raises(ValueError):
            agma.size_factor(0)

    def test_elastic_coefficient_steel(self):
        """ZE for steel on steel is about 190 sqrt(MPa)."""
        from mecapy.materials import get_material_properties

        steel = get_material_properties("steel")
        ze = agma.elastic_coefficient(steel, steel)
        assert ze == pytest.approx(190.0, rel=0.02)

    def test_geometry_factor_I(self):
        """Norton I for a 20 deg 17/52 spur mesh (Eq. 12.22)."""
        pinion = SpurGear(17, module=2.54)
        gear = SpurGear(52, module=2.54)
        zi = agma.geometry_factor_I(pinion, gear)
        assert zi == pytest.approx(0.0986, abs=0.001)
        # Dimensionless: independent of module.
        assert zi == pytest.approx(
            agma.geometry_factor_I(SpurGear(17, module=1.0),
                                   SpurGear(52, module=1.0)))

    def test_geometry_factor_I_rack(self):
        """Rack mesh (gear=None) drops the gear curvature term."""
        pinion = SpurGear(17, module=2.54)
        gear = SpurGear(52, module=2.54)
        zi_rack = agma.geometry_factor_I(pinion)
        assert zi_rack > agma.geometry_factor_I(pinion, gear)

    def test_life_factors_at_1e7(self):
        """YN and ZN are about 1.0 at 10^7 cycles."""
        assert agma.bending_life_factor(1e7) == pytest.approx(1.0, abs=0.02)
        assert agma.contact_life_factor(1e7) == pytest.approx(1.0, abs=0.02)

    def test_reliability_factor(self):
        """Table values and range check."""
        assert agma_data.reliability_factor(0.99) == pytest.approx(1.0)
        assert agma_data.reliability_factor(0.999) == pytest.approx(1.25)
        assert agma_data.reliability_factor(0.90) == pytest.approx(0.85)
        with pytest.raises(ValueError):
            agma_data.reliability_factor(0.3)

    def test_hardness_ratio_factor(self):
        """ZW is 1 below ratio 1.2 and grows with mG above it."""
        assert agma.hardness_ratio_factor(300, 300, 3.0) == 1.0
        assert agma.hardness_ratio_factor(450, 300, 3.0) > 1.0

    def test_temperature_factor(self):
        """Y_theta is 1 below 110 C, continuous at 110, grows above."""
        assert agma.temperature_factor(90) == 1.0
        assert agma.temperature_factor(110) == pytest.approx(1.0)
        assert agma.temperature_factor(230) == pytest.approx(1.3636, rel=1e-3)
        with pytest.raises(ValueError):
            agma.temperature_factor(-300)

    def test_allowable_stress_fits(self):
        """Grade 1 fits at HB 300."""
        st = agma_data.allowable_bending_stress(300, grade=1)
        assert st == pytest.approx(0.533 * 300 + 88.3)
        sc = agma_data.allowable_contact_stress(300, grade=1)
        assert sc == pytest.approx(2.22 * 300 + 200.0)
        with pytest.raises(ValueError):
            agma_data.allowable_bending_stress(300, grade=3)

    def test_geometry_factor_J(self):
        """Digitized Fig. 14-6 anchors: 17T/52T mesh."""
        assert agma_data.geometry_factor_J(17, 52) == pytest.approx(0.30,
                                                                    abs=0.01)
        assert agma_data.geometry_factor_J(52, 17) == pytest.approx(0.40,
                                                                    abs=0.01)
        with pytest.raises(ValueError):
            agma_data.geometry_factor_J(10, 40)

    def test_lewis_form_factor(self):
        """Table anchors and rack limit."""
        assert agma_data.lewis_form_factor(20) == pytest.approx(0.322)
        assert agma_data.lewis_form_factor(500) == pytest.approx(0.485)
        with pytest.raises(ValueError):
            agma_data.lewis_form_factor(10)


class TestShigleyExample:
    """End-to-end validation against the Shigley 17T/52T spur example."""

    @pytest.fixture
    def rating(self):
        pinion = SpurGear(17, diametral_pitch=10, face_width=38.1)
        gear = SpurGear(52, diametral_pitch=10, face_width=38.1)
        return AGMARating(pinion, gear, power_kw=2.9828,
                          pinion_speed_rpm=1800, Qv=6, hardness_HB=240)

    def test_tangential_force(self, rating):
        """Wt about 164.8 lbf = 733 N."""
        assert rating.Ft == pytest.approx(733.0, rel=0.01)

    def test_intermediate_factors(self, rating):
        """Kv, KH per Shigley; ZI per Norton Eq. 12.22."""
        assert rating.Kv == pytest.approx(1.377, rel=0.01)
        assert rating.KH == pytest.approx(1.22, rel=0.01)
        assert rating.ZI == pytest.approx(0.0986, rel=0.01)

    def test_bending_stress(self, rating):
        """sigma_F about 6.15 kpsi = 42.4 MPa (5%: digitized J)."""
        assert rating.bending_stress_pinion == pytest.approx(42.4, rel=0.05)

    def test_contact_stress(self, rating):
        """Shigley's 475 MPa scaled by sqrt(0.121 / 0.0986) for
        Norton's I: about 526 MPa (5%)."""
        assert rating.contact_stress == pytest.approx(526.0, rel=0.05)

    def test_safety_factors_positive(self, rating):
        """Safety factors and allowables are computed and positive."""
        assert rating.SF_pinion > 0
        assert rating.SF_gear > 0
        assert rating.SH > 0
        assert rating.allowable_bending_stress > 0

    def test_summary(self, rating):
        """Summary mentions the key results."""
        text = rating.summary()
        assert "Bending stress" in text
        assert "Contact stress" in text
        assert "SH" in text


class TestRatingValidation:
    """Input validation of AGMARating."""

    def test_face_width_required(self):
        pinion = SpurGear(17, module=2.5)
        gear = SpurGear(52, module=2.5)
        with pytest.raises(ValueError):
            AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800,
                       hardness_HB=240)

    def test_load_specification(self):
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        with pytest.raises(ValueError):
            AGMARating(pinion, gear, pinion_speed_rpm=1800, hardness_HB=240)
        with pytest.raises(ValueError):
            AGMARating(pinion, gear, power_kw=3.0, tangential_force=700.0,
                       pinion_speed_rpm=1800, hardness_HB=240)

    def test_ki_defaults_to_one_and_scales_bending_stress(self):
        """Ki_pinion/Ki_gear default to 1.0; a non-default value scales
        only its own member's bending stress, not the other member's.
        """
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        base = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800)
        assert base.Ki_pinion == 1.0
        assert base.Ki_gear == 1.0

        idler_pinion = AGMARating(pinion, gear, power_kw=3.0,
                                  pinion_speed_rpm=1800, Ki_pinion=1.42)
        assert idler_pinion.bending_stress_pinion == pytest.approx(
            base.bending_stress_pinion * 1.42)
        assert idler_pinion.bending_stress_gear == pytest.approx(
            base.bending_stress_gear)

    def test_ki_validation(self):
        """Ki_pinion/Ki_gear must be strictly positive."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        with pytest.raises(ValueError):
            AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800,
                       Ki_pinion=0)
        with pytest.raises(ValueError):
            AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800,
                       Ki_gear=-1)

    def test_allowables_required_only_when_used(self):
        """Without material data the stresses work; the strengths raise."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        r = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800)
        assert r.has_allowables is False
        assert r.bending_stress_pinion > 0
        assert r.contact_stress > 0
        for attr in ("St", "Sc", "SF_pinion", "SH"):
            with pytest.raises(ValueError):
                getattr(r, attr)

    def test_required_st_range(self):
        """max_safety_factor turns required_St into a (min, max) range."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        r = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800)
        low, high = r.required_St(1.2, max_safety_factor=2.0)
        assert low == pytest.approx(r.required_St(1.2))
        assert high == pytest.approx(r.required_St(2.0))
        assert low < high

    def test_required_st_range_validation(self):
        """max_safety_factor must exceed safety_factor."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        r = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800)
        with pytest.raises(ValueError):
            r.required_St(1.5, max_safety_factor=1.5)
        with pytest.raises(ValueError):
            r.required_St(1.5, max_safety_factor=1.0)

    def test_required_sc_range(self):
        """max_safety_factor turns required_Sc into a (min, max) range."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        r = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800)
        low, high = r.required_Sc(1.2, max_safety_factor=2.0)
        assert low == pytest.approx(r.required_Sc(1.2))
        assert high == pytest.approx(r.required_Sc(2.0))
        assert low < high

    def test_required_sc_range_validation(self):
        """max_safety_factor must exceed safety_factor."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        r = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800)
        with pytest.raises(ValueError):
            r.required_Sc(1.5, max_safety_factor=1.5)

    def test_required_strengths(self):
        """required_strengths bundles required_St and required_Sc."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        r = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800)

        both = r.required_strengths(1.3)
        assert both == {
            "St": pytest.approx(r.required_St(1.3)),
            "Sc": pytest.approx(r.required_Sc(1.3)),
        }

        ranged = r.required_strengths(1.2, max_safety_factor=2.0)
        assert ranged["St"] == pytest.approx(r.required_St(1.2, max_safety_factor=2.0))
        assert ranged["Sc"] == pytest.approx(r.required_Sc(1.2, max_safety_factor=2.0))

    def test_explicit_allowables(self):
        """Named gear-steel entries can feed St/Sc directly."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        steel = agma_data.GEAR_STEELS["carburized_grade1"]
        rating = AGMARating(pinion, gear, power_kw=3.0,
                            pinion_speed_rpm=1800,
                            St=steel["St"], Sc=steel["Sc"])
        assert rating.St == steel["St"]
        assert rating.SH > 0

    def test_temperature_celsius(self):
        """temperature_celsius derates the allowables via Y_theta."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        base = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800,
                          hardness_HB=240)
        hot = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800,
                         hardness_HB=240, temperature_celsius=200)
        assert hot.temperature_factor == pytest.approx(agma.temperature_factor(200))
        assert hot.allowable_bending_stress < base.allowable_bending_stress
        assert hot.allowable_contact_stress < base.allowable_contact_stress
        # Giving both temperature inputs is an error.
        with pytest.raises(ValueError):
            AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800,
                       hardness_HB=240, temperature_celsius=200,
                       temperature_factor=1.2)

    def test_rack_mesh(self):
        """A pinion on a rack rates via the rack J column."""
        pinion = SpurGear(20, module=2.0, face_width=25.0)
        rack = Rack(module=2.0, face_width=25.0)
        rating = AGMARating(pinion, rack, power_kw=1.0,
                            pinion_speed_rpm=600, hardness_HB=240)
        assert rating.bending_stress_pinion > 0
        assert rating.bending_stress_gear is None
        assert rating.SF_gear is None

    def test_helical_rating(self):
        """A helical mesh rates with helical J and mN < 1... > 0 checks."""
        pinion = HelicalGear(20, module=3.0, helix_angle=20.0, hand="right",
                             face_width=40.0)
        gear = HelicalGear(60, module=3.0, helix_angle=20.0, hand="left",
                           face_width=40.0)
        rating = AGMARating(pinion, gear, power_kw=10.0,
                            pinion_speed_rpm=1200, hardness_HB=300)
        assert rating.bending_stress_pinion > 0
        assert rating.contact_stress > 0
        # Helical ZI exceeds the spur value thanks to the larger
        # transverse pressure angle (Norton I in the transverse plane).
        spur_zi = agma.geometry_factor_I(SpurGear(20, module=3.0),
                                         SpurGear(60, module=3.0))
        assert rating.ZI > spur_zi


class TestRatingIsLazy:
    """D3: the rating recomputes from its inputs and never caches (Phase 4)."""

    def _rating(self):
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        return AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800,
                          hardness_HB=240), pinion, gear

    def test_safety_factor_recomputes_on_face_width_change(self):
        """Narrowing the governing (narrower) member lowers the SFs."""
        rating, pinion, gear = self._rating()
        sf0, sh0 = rating.SF_pinion, rating.SH
        gear.face_width = 25.0  # now the narrower member governs b
        assert rating.face_width == 25.0
        assert rating.SF_pinion < sf0
        assert rating.SH < sh0

    def test_stress_recomputes_on_overload_change(self):
        """Raising the overload factor Ko raises the bending stress."""
        rating, _, _ = self._rating()
        sigma0 = rating.bending_stress_pinion
        rating.Ko = 1.5
        assert rating.bending_stress_pinion == pytest.approx(1.5 * sigma0)

    def test_ks_follows_the_module(self):
        """Ks comes from the module and tracks a change to it."""
        rating, pinion, _ = self._rating()
        assert rating.Ks == 1.00                    # module 2.5
        pinion.module = 8.0
        assert rating.Ks == 1.15

    def test_ks_override_wins(self):
        """An explicit Ks bypasses the module table."""
        pinion = SpurGear(17, module=10.0, face_width=40.0)
        gear = SpurGear(52, module=10.0, face_width=40.0)
        rating = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800,
                            hardness_HB=240, Ks=1.0)
        assert rating.Ks == 1.0                     # table would give 1.25
        rating.Ks = None                            # back to the table
        assert rating.Ks == 1.25
        with pytest.raises(ValueError):
            rating.Ks = 0

    def test_ks_raises_the_stress(self):
        """A coarse-module mesh is penalised by Ks."""
        fine = AGMARating(SpurGear(17, module=4.0, face_width=40.0),
                          SpurGear(52, module=4.0, face_width=40.0),
                          tangential_force=1000.0, pinion_speed_rpm=100,
                          hardness_HB=240)
        coarse = AGMARating(SpurGear(17, module=10.0, face_width=40.0),
                            SpurGear(52, module=10.0, face_width=40.0),
                            tangential_force=1000.0, pinion_speed_rpm=100,
                            hardness_HB=240)
        assert fine.Ks == 1.00 and coarse.Ks == 1.25
        assert coarse._common_load > fine._common_load

    def test_yj_override_used_verbatim(self):
        """A YJ override is returned as-is, not recomputed."""
        pinion = SpurGear(17, module=2.5, face_width=40.0)
        gear = SpurGear(52, module=2.5, face_width=40.0)
        rating = AGMARating(pinion, gear, power_kw=3.0, pinion_speed_rpm=1800,
                            hardness_HB=240, YJ_pinion=0.42, YJ_gear=0.48)
        assert rating.YJ_pinion == 0.42
        assert rating.YJ_gear == 0.48


class TestGearMaterial:
    """A Material subclass carrying Brinell hardness and AGMA grade."""

    def test_construction_and_defaults(self):
        """Hardness and grade are stored; grade defaults to 1."""
        m = GearMaterial(hardness_HB=240)
        assert m.hardness_HB == 240
        assert m.grade == 1
        assert isinstance(m, Material)   # base.py stress/safety dispatch

    def test_forwards_material_kwargs(self):
        """Extra kwargs flow through to the Material constructor."""
        m = GearMaterial(240, base_material="steel", yield_strength=600,
                         ultimate_strength=900, name="alloy")
        assert m.yield_strength == 600
        assert m.name == "alloy"
        assert m.properties_si["yield_strength"] == 600e6

    def test_allowables_match_agma_data(self):
        """St/Sc properties equal the agma_data through-hardened fits."""
        m = GearMaterial(240, grade=2)
        assert m.allowable_bending_stress == pytest.approx(
            agma_data.allowable_bending_stress(240, 2))
        assert m.allowable_contact_stress == pytest.approx(
            agma_data.allowable_contact_stress(240, 2))

    def test_recompute_on_hardness_change(self):
        """Changing the hardness updates the derived allowables."""
        m = GearMaterial(240, grade=1)
        before = m.allowable_bending_stress
        m.hardness_HB = 300
        assert m.allowable_bending_stress > before
        assert m.allowable_bending_stress == pytest.approx(
            agma_data.allowable_bending_stress(300, 1))

    def test_invalid_hardness(self):
        with pytest.raises(ValueError):
            GearMaterial(hardness_HB=0)
        m = GearMaterial(240)
        with pytest.raises(ValueError):
            m.hardness_HB = -10

    def test_invalid_grade(self):
        with pytest.raises(ValueError):
            GearMaterial(240, grade=3)

    def test_agma_pulls_hardness_and_grade(self):
        """AGMARating reads hardness/grade off the gear materials."""
        pinion = SpurGear(20, module=3.0, face_width=40.0,
                          material=GearMaterial(240, grade=1))
        gear = SpurGear(60, module=3.0, face_width=40.0,
                        material=GearMaterial(200, grade=1))
        auto = AGMARating(pinion, gear, power_kw=10.0, pinion_speed_rpm=1200)
        explicit = AGMARating(pinion, gear, power_kw=10.0,
                              pinion_speed_rpm=1200, hardness_HB=240,
                              gear_hardness_HB=200, grade=1)
        assert auto.St == pytest.approx(explicit.St)
        assert auto.Sc == pytest.approx(explicit.Sc)
        assert auto.ZW == pytest.approx(explicit.ZW)
        assert auto.ZW > 1.0   # 240/200 hardness ratio enables ZW

    def test_explicit_hardness_overrides_material(self):
        """An explicit hardness_HB still wins over the material's value."""
        pinion = SpurGear(20, module=3.0, face_width=40.0,
                          material=GearMaterial(240, grade=1))
        gear = SpurGear(60, module=3.0, face_width=40.0)
        r = AGMARating(pinion, gear, power_kw=10.0, pinion_speed_rpm=1200,
                       hardness_HB=300)
        assert r.St == pytest.approx(agma_data.allowable_bending_stress(300, 1))

    def test_string_material_still_requires_hardness(self):
        """A plain string material has no hardness to fall back on."""
        pinion = SpurGear(20, module=3.0, face_width=40.0, material="steel")
        gear = SpurGear(60, module=3.0, face_width=40.0)
        r = AGMARating(pinion, gear, power_kw=10.0, pinion_speed_rpm=1200)
        assert r.has_allowables is False
        with pytest.raises(ValueError, match="hardness_HB"):
            _ = r.SF_pinion


class TestInternalMeshRating:
    """AGMA rating with an internal (ring) gear as the mating member.

    These ratings are approximate (the J tables are external-tooth
    data); what is checked here is the internal contact geometry and
    the guards around it.
    """

    def test_internal_geometry_factor_is_larger(self):
        """A concave ring flank lowers the contact stress."""
        pinion = SpurGear(20, module=2.0, face_width=25.0)
        external = SpurGear(80, module=2.0, face_width=25.0)
        ring = SpurGear(80, module=2.0, face_width=25.0, internal=True)
        assert (agma.geometry_factor_I(pinion, ring)
                > agma.geometry_factor_I(pinion, external))

    def test_matches_the_pitch_point_closed_form(self):
        """I_internal / I_external tends to (mG + 1) / (mG - 1).

        The implementation samples the curvature one base pitch inside
        the pinion tip, so the closed-form ratio is only reached in the
        pitch-point limit — approached here with a very large pinion,
        for which that offset is negligible.
        """
        pinion = SpurGear(1000, module=1.0, face_width=25.0)
        external = SpurGear(3000, module=1.0, face_width=25.0)
        ring = SpurGear(3000, module=1.0, face_width=25.0, internal=True)
        ratio = (agma.geometry_factor_I(pinion, ring)
                 / agma.geometry_factor_I(pinion, external))
        mG = 3.0
        assert ratio == pytest.approx((mG + 1) / (mG - 1), rel=1e-3)

    def test_rating_is_finite_and_gentler_than_external(self):
        """The same load in a ring gives a higher contact safety factor."""
        pinion = SpurGear(20, module=2.0, face_width=25.0)
        ring = SpurGear(80, module=2.0, face_width=25.0, internal=True)
        external = SpurGear(80, module=2.0, face_width=25.0)
        internal_rating = AGMARating(pinion, ring, power_kw=5.0,
                                     pinion_speed_rpm=1200.0,
                                     hardness_HB=250)
        external_rating = AGMARating(pinion, external, power_kw=5.0,
                                     pinion_speed_rpm=1200.0,
                                     hardness_HB=250)
        assert internal_rating.SH > external_rating.SH
        assert internal_rating.SF_pinion > 0
        assert internal_rating.SF_gear > 0

    def test_hardness_ratio_factor_is_one(self):
        """ZW is defined for external meshes only."""
        pinion = SpurGear(20, module=2.0, face_width=25.0)
        ring = SpurGear(80, module=2.0, face_width=25.0, internal=True)
        rating = AGMARating(pinion, ring, power_kw=5.0,
                            pinion_speed_rpm=1200.0, hardness_HB=300,
                            gear_hardness_HB=200)
        assert rating.is_internal_mesh is True
        assert rating.ZW == 1.0

    def test_ring_cannot_be_the_pinion(self):
        """The first argument must be the external member."""
        pinion = SpurGear(20, module=2.0, face_width=25.0)
        ring = SpurGear(80, module=2.0, face_width=25.0, internal=True)
        with pytest.raises(ValueError, match="external member"):
            AGMARating(ring, pinion, power_kw=5.0, pinion_speed_rpm=300.0)

    def test_train_rating_orders_the_stage(self):
        """rate_agma picks the external member as the pinion either way."""
        from mecapy.gears import Transmission

        ring = SpurGear(80, module=2.0, face_width=25.0, internal=True,
                        speed_rpm=300.0, power_kw=5.0)
        pinion = SpurGear(20, module=2.0, face_width=25.0)
        ratings = Transmission().add_stage(ring, pinion).rate_agma(
            hardness_HB=250)
        assert len(ratings) == 1
        assert ratings[0].pinion is pinion
        assert ratings[0].gear is ring
