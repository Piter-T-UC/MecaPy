"""Tests for the Material class (fatigue analysis, Shigley Ch. 6)."""

import math

import pytest

from mecapy import MechaElement
from mecapy.materials import Material, add_custom_material
from mecapy.utils import constants


class TestMaterialConstruction:
    """Test cases for Material construction and input validation."""

    def test_database_defaults_in_mpa(self):
        """Strengths default from the steel database row, converted to MPa."""
        m = Material("steel")
        assert m.ultimate_strength == pytest.approx(400.0)
        assert m.yield_strength == pytest.approx(250.0)

    def test_explicit_override_wins(self):
        """An explicit ultimate_strength overrides the database default."""
        m = Material("steel", ultimate_strength=690.0)
        assert m.ultimate_strength == pytest.approx(690.0)
        assert m.yield_strength == pytest.approx(250.0)

    def test_unknown_base_material_raises(self):
        """An unknown base material raises ValueError."""
        with pytest.raises(ValueError):
            Material("unobtainium")

    def test_missing_ultimate_strength_raises(self):
        """A custom database row without Sut requires an explicit value."""
        add_custom_material("no_sut", {"yield_strength": 100e6})
        try:
            with pytest.raises(ValueError):
                Material("no_sut")
            m = Material("no_sut", ultimate_strength=200.0)
            assert m.ultimate_strength == pytest.approx(200.0)
        finally:
            del constants.MATERIALS["no_sut"]

    def test_nonpositive_ultimate_strength_raises(self):
        """A non-positive Sut raises ValueError."""
        with pytest.raises(ValueError):
            Material("steel", ultimate_strength=-100)

    def test_yield_above_ultimate_raises(self):
        """Sy at or above Sut raises ValueError."""
        with pytest.raises(ValueError):
            Material("steel", ultimate_strength=400, yield_strength=400)

    def test_unknown_surface_finish_raises(self):
        """An unknown surface finish raises ValueError."""
        with pytest.raises(ValueError):
            Material("steel", surface_finish="polished")

    def test_unknown_loading_raises(self):
        """An unknown loading type raises ValueError."""
        with pytest.raises(ValueError):
            Material("steel", loading="shear")

    def test_reliability_out_of_range_raises(self):
        """A reliability outside [0.5, 1) raises ValueError."""
        with pytest.raises(ValueError):
            Material("steel", reliability=1.0)
        with pytest.raises(ValueError):
            Material("steel", reliability=0.4)

    def test_reliability_off_table_is_computed(self):
        """A reliability not in Table 6-5 gets ke from Shigley Eq. 6-30."""
        # R=0.98 sits between the 0.95 (0.868) and 0.99 (0.814) table rows.
        ke = Material("steel", reliability=0.98).ke
        assert 0.814 < ke < 0.868

    def test_temperature_out_of_range_raises(self):
        """A temperature outside the Table 6-4 range raises ValueError."""
        with pytest.raises(ValueError):
            Material("steel", temperature=700)

    def test_diameter_out_of_range_raises(self):
        """A diameter outside the Eq. 6-20 validity range raises ValueError."""
        with pytest.raises(ValueError):
            Material("steel", diameter=300)
        with pytest.raises(ValueError):
            Material("steel", diameter=1.0)

    def test_repr(self):
        """__repr__ shows the identifying constructor args."""
        m = Material("steel")
        assert "steel" in repr(m)
        assert "400" in repr(m)
        assert "Alloy A" in repr(Material("steel", name="Alloy A"))


class TestMarinFactors:
    """Test cases for the Marin factors and the endurance limit."""

    def test_ka_machined(self):
        """ka for machined steel is 4.51 * Sut^-0.265."""
        m = Material("steel")
        assert m.ka == pytest.approx(4.51 * 400 ** -0.265)

    def test_ka_ground(self):
        """ka for a ground finish is 1.58 * Sut^-0.085."""
        m = Material("steel", surface_finish="ground")
        assert m.ka == pytest.approx(1.58 * 400 ** -0.085)

    def test_kb_small_diameter(self):
        """kb for d <= 51 mm follows (d/7.62)^-0.107."""
        m = Material("steel", diameter=25)
        assert m.kb == pytest.approx((25 / 7.62) ** -0.107)

    def test_kb_large_diameter(self):
        """kb for d > 51 mm follows 1.51 * d^-0.157."""
        m = Material("steel", diameter=100)
        assert m.kb == pytest.approx(1.51 * 100 ** -0.157)

    def test_kb_axial_is_one(self):
        """Axial loading gives kb = 1 regardless of diameter."""
        m = Material("steel", diameter=100, loading="axial")
        assert m.kb == 1.0

    def test_kb_no_diameter_is_one(self):
        """Without a diameter, kb defaults to 1."""
        assert Material("steel").kb == 1.0

    def test_eq_diameter_round_trip(self):
        """eq_diameter recovers d from a round section's 95% area."""
        m = Material("steel")
        d_e = m.eq_diameter(0.0766 * 50 ** 2)  # A_0.95sigma for d = 50 mm
        assert d_e == pytest.approx(50.0)
        assert m.diameter == pytest.approx(50.0)  # it also set the diameter
        assert m.kb == pytest.approx((50 / 7.62) ** -0.107)

    def test_eq_diameter_invalid_area(self):
        """A non-positive 95% area raises ValueError."""
        with pytest.raises(ValueError):
            Material("steel").eq_diameter(0)

    def test_eq_diameter_out_of_kb_range(self):
        """A d_e outside the kb validity range raises ValueError."""
        with pytest.raises(ValueError):
            Material("steel").eq_diameter(5000)  # d_e ~ 255 mm > 254

    @pytest.mark.parametrize("loading,expected",
                             [("bending", 1.0), ("axial", 0.85),
                              ("torsion", 0.59)])
    def test_kc_values(self, loading, expected):
        """kc matches the Eq. 6-26 load factors."""
        assert Material("steel", loading=loading).kc == expected

    def test_kd_table_point(self):
        """kd at an exact Table 6-4 temperature returns the table value."""
        assert Material("steel", temperature=150).kd == pytest.approx(1.025)

    def test_kd_interpolated(self):
        """kd interpolates linearly between table temperatures."""
        assert Material("steel", temperature=75).kd == pytest.approx(1.015)

    def test_ke_table_value(self):
        """ke at 99% reliability is 0.814 (Table 6-5)."""
        assert Material("steel", reliability=0.99).ke == pytest.approx(0.814)

    def test_se_prime_low_sut(self):
        """Se' is 0.5 * Sut for Sut <= 1400 MPa."""
        assert Material("steel").se_prime == pytest.approx(200.0)

    def test_se_prime_high_sut(self):
        """Se' caps at 700 MPa for Sut > 1400 MPa."""
        m = Material("steel", ultimate_strength=1600, yield_strength=1300)
        assert m.se_prime == pytest.approx(700.0)

    def test_endurance_limit_is_marin_product(self):
        """Se equals ka*kb*kc*kd*ke*Se' (Marin equation)."""
        m = Material("steel", surface_finish="ground", diameter=25,
                     temperature=150, reliability=0.99, loading="torsion")
        expected = m.ka * m.kb * m.kc * m.kd * m.ke * m.se_prime
        assert m.endurance_limit == pytest.approx(expected)


class TestRecompute:
    """Tests protecting the recompute-don't-cache guarantee."""

    def test_changing_finish_updates_ka_and_se(self):
        """Changing surface_finish after construction updates ka and Se."""
        m = Material("steel")
        se_machined = m.endurance_limit
        m.surface_finish = "ground"
        assert m.ka == pytest.approx(1.58 * 400 ** -0.085)
        assert m.endurance_limit != pytest.approx(se_machined)

    def test_changing_diameter_updates_kb(self):
        """Changing diameter after construction updates kb."""
        m = Material("steel", diameter=25)
        kb_small = m.kb
        m.diameter = 100
        assert m.kb == pytest.approx(1.51 * 100 ** -0.157)
        assert m.kb != pytest.approx(kb_small)

    def test_setters_revalidate(self):
        """Setting an invalid value after construction still raises."""
        m = Material("steel")
        with pytest.raises(ValueError):
            m.ultimate_strength = -1
        with pytest.raises(ValueError):
            m.surface_finish = "polished"
        with pytest.raises(ValueError):
            m.diameter = 500


class TestSNCurve:
    """Test cases for the S-N (Woehler) curve."""

    def test_f_low_sut(self):
        """The fatigue-strength fraction f is 0.9 below 482.6 MPa."""
        assert Material("steel").f == pytest.approx(0.9)

    def test_sn_coefficients(self):
        """sn_a and sn_b follow Eq. 6-14 / 6-15."""
        m = Material("steel")
        f_sut = m.f * m.ultimate_strength
        assert m.sn_a == pytest.approx(f_sut ** 2 / m.endurance_limit)
        assert m.sn_b == pytest.approx(
            -math.log10(f_sut / m.endurance_limit) / 3)

    def test_cycles_at_f_sut_is_1000(self):
        """Fully reversed stress of f*Sut fails at 10^3 cycles."""
        m = Material("steel")
        n = m.cycles_to_failure(m.f * m.ultimate_strength)
        assert n == pytest.approx(1e3)

    def test_below_endurance_limit_is_infinite(self):
        """A stress below Se gives infinite life."""
        m = Material("steel")
        assert m.cycles_to_failure(m.endurance_limit * 0.9) == math.inf

    def test_mean_stress_shortens_life(self):
        """A tensile mean stress reduces the cycles to failure."""
        m = Material("steel")
        assert (m.cycles_to_failure(250, mean_stress=100)
                < m.cycles_to_failure(250))

    def test_mean_stress_at_sut_raises(self):
        """A mean stress at or above Sut raises ValueError."""
        m = Material("steel")
        with pytest.raises(ValueError):
            m.cycles_to_failure(100, mean_stress=400)


class TestLoadCases:
    """Test cases for load cases, Goodman and Miner's rule."""

    def test_goodman_safety_factor(self):
        """The Goodman safety factor follows Eq. 6-46."""
        m = Material("steel")
        sf = m.goodman_safety_factor(100, mean_stress=50)
        expected = 1 / (100 / m.endurance_limit + 50 / m.ultimate_strength)
        assert sf == pytest.approx(expected)

    def test_add_load_case_chains(self):
        """add_load_case returns self so calls chain."""
        m = Material("steel")
        result = m.add_load_case(200, 100, cycles=1e4).add_load_case(150, 0)
        assert result is m
        assert len(m.load_cases) == 2

    def test_invalid_load_case_raises(self):
        """Invalid amplitude, mean stress or cycles raise ValueError."""
        m = Material("steel")
        with pytest.raises(ValueError):
            m.add_load_case(-10)
        with pytest.raises(ValueError):
            m.add_load_case(100, mean_stress=400)
        with pytest.raises(ValueError):
            m.add_load_case(100, cycles=0)

    def test_edit_by_label(self):
        """edit_load_case by label updates and revalidates the case."""
        m = Material("steel").add_load_case(200, 100, cycles=1e4,
                                            label="startup")
        m.edit_load_case("startup", stress_amplitude=250)
        assert m.load_cases[0]["stress_amplitude"] == 250
        with pytest.raises(ValueError):
            m.edit_load_case("startup", mean_stress=500)

    def test_edit_unknown_field_raises(self):
        """Editing an unknown load-case field raises ValueError."""
        m = Material("steel").add_load_case(200)
        with pytest.raises(ValueError):
            m.edit_load_case(0, frequency=50)

    def test_remove_by_index_and_label(self):
        """remove_load_case works by position and by label."""
        m = (Material("steel").add_load_case(200, label="a")
             .add_load_case(150, label="b"))
        m.remove_load_case("a")
        assert len(m.load_cases) == 1
        m.remove_load_case(0)
        assert m.load_cases == []

    def test_missing_case_raises(self):
        """An unknown label or out-of-range index raises ValueError."""
        m = Material("steel")
        with pytest.raises(ValueError):
            m.remove_load_case("ghost")
        with pytest.raises(ValueError):
            m.edit_load_case(3, cycles=10)

    def test_miner_damage_hand_sum(self):
        """miner_damage equals the hand-summed n_i / N_i."""
        m = (Material("steel").add_load_case(300, cycles=1e4)
             .add_load_case(250, 50, cycles=5e4))
        expected = (1e4 / m.cycles_to_failure(300)
                    + 5e4 / m.cycles_to_failure(250, 50))
        assert m.miner_damage() == pytest.approx(expected)

    def test_remaining_life_inverse_of_damage(self):
        """remaining_life is 1 / miner_damage for finite damage."""
        m = Material("steel").add_load_case(300, cycles=1e4)
        assert m.remaining_life() == pytest.approx(1 / m.miner_damage())

    def test_infinite_life_cases_contribute_zero(self):
        """Cases below the endurance limit give zero damage, infinite life."""
        m = Material("steel")
        m.add_load_case(m.endurance_limit * 0.5, cycles=1e9)
        assert m.miner_damage() == 0
        assert m.remaining_life() == math.inf

    def test_damage_recomputes_after_finish_change(self):
        """Changing the surface finish changes the damage of stored cases."""
        m = Material("steel").add_load_case(300, cycles=1e4)
        damage_machined = m.miner_damage()
        m.surface_finish = "as_forged"
        assert m.miner_damage() != pytest.approx(damage_machined)


class TestMeanStressCriteria:
    """Test cases for the Soderberg/Gerber/ASME/Langer criteria."""

    def test_each_criterion_positive(self):
        """Every fatigue criterion returns a finite positive nf."""
        m = Material("steel")  # Sut=400, Sy=250 MPa
        for c in ("goodman", "soderberg", "gerber", "asme_elliptic"):
            nf = m.fatigue_safety_factor(100, 80, criterion=c)
            assert math.isfinite(nf) and nf > 0

    def test_conservatism_ordering(self):
        """Soderberg <= Goodman <= Gerber for tensile mean stress."""
        m = Material("steel")
        sod = m.fatigue_safety_factor(100, 80, "soderberg")
        good = m.fatigue_safety_factor(100, 80, "goodman")
        ger = m.fatigue_safety_factor(100, 80, "gerber")
        assert sod <= good <= ger

    def test_goodman_alias(self):
        """goodman_safety_factor equals the goodman fatigue factor."""
        m = Material("steel")
        assert (m.goodman_safety_factor(100, 50)
                == m.fatigue_safety_factor(100, 50, "goodman"))

    def test_soderberg_formula(self):
        """Soderberg nf follows Eq. 6-45 (uses Sy, not Sut)."""
        m = Material("steel")
        nf = m.fatigue_safety_factor(100, 60, "soderberg")
        expected = 1 / (100 / m.endurance_limit + 60 / m.yield_strength)
        assert nf == pytest.approx(expected)

    def test_asme_elliptic_formula(self):
        """ASME-elliptic nf follows Eq. 6-50."""
        m = Material("steel")
        nf = m.fatigue_safety_factor(100, 60, "asme_elliptic")
        expected = 1 / math.sqrt((100 / m.endurance_limit) ** 2
                                 + (60 / m.yield_strength) ** 2)
        assert nf == pytest.approx(expected)

    def test_cycles_differ_by_criterion(self):
        """Each criterion folds mean stress differently -> different N."""
        m = Material("steel")
        n_good = m.cycles_to_failure(200, 100, "goodman")
        n_sod = m.cycles_to_failure(200, 100, "soderberg")
        assert n_sod < n_good  # Soderberg folds to a higher sigma_ar

    def test_cycles_default_is_goodman(self):
        """cycles_to_failure defaults to the goodman fold (unchanged)."""
        m = Material("steel")
        assert (m.cycles_to_failure(200, 100)
                == m.cycles_to_failure(200, 100, "goodman"))

    def test_langer_safety_factor(self):
        """Langer yield factor follows Eq. 6-49, n = Sy/(sa+sm)."""
        m = Material("steel")
        assert (m.langer_safety_factor(100, 50)
                == pytest.approx(m.yield_strength / 150))

    def test_governing_picks_yield(self):
        """Large mean stress -> Langer yield governs."""
        m = Material("steel")
        nf, mode = m.governing_safety_factor(5, 240)
        assert mode == "yield"
        assert nf == pytest.approx(m.langer_safety_factor(5, 240))

    def test_governing_picks_fatigue(self):
        """Large amplitude -> the fatigue criterion governs."""
        m = Material("steel")
        nf, mode = m.governing_safety_factor(180, 5)
        assert mode == "fatigue"
        assert nf == pytest.approx(m.fatigue_safety_factor(180, 5))

    def test_unknown_criterion_raises(self):
        """An unknown criterion raises ValueError."""
        m = Material("steel")
        with pytest.raises(ValueError):
            m.fatigue_safety_factor(100, 50, "junk")

    def test_soderberg_mean_above_yield_raises(self):
        """Soderberg/ASME reject sigma_m >= Sy (out of their range)."""
        m = Material("steel")  # Sy = 250 MPa
        with pytest.raises(ValueError):
            m.fatigue_safety_factor(100, 250, "soderberg")
        with pytest.raises(ValueError):
            m.cycles_to_failure(100, 260, "asme_elliptic")


class TestMechaElementIntegration:
    """Test cases for using a Material inside MechaElement."""

    def test_material_object_properties_in_pa(self):
        """material_properties returns Pa values from a Material object."""
        elem = MechaElement(material=Material("steel"))
        props = elem.material_properties
        assert props["yield_strength"] == pytest.approx(250e6)
        assert props["ultimate_strength"] == pytest.approx(400e6)
        assert props["elastic_modulus"] == pytest.approx(210e9)

    def test_overridden_strengths_propagate(self):
        """Explicit MPa overrides show up in Pa for element consumers."""
        elem = MechaElement(material=Material("steel", yield_strength=300))
        assert elem.material_properties["yield_strength"] == pytest.approx(300e6)

    def test_safety_factor_with_material_object(self):
        """The inherited yield safety factor works with a Material object."""
        elem = MechaElement(material=Material("steel"))
        assert elem.safety_factor(125e6) == pytest.approx(2.0)

    def test_string_material_still_works(self):
        """String material names keep working unchanged (regression)."""
        elem = MechaElement(material="steel")
        assert elem.material_properties["yield_strength"] == pytest.approx(250e6)


class TestPlots:
    """Smoke tests for the matplotlib plots."""

    def test_plot_sn_curve_returns_figure(self):
        """plot_sn_curve draws without showing and returns a Figure."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m = Material("steel").add_load_case(300, cycles=1e4, label="startup")
        fig = m.plot_sn_curve(show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_goodman_returns_figure(self):
        """plot_goodman draws without showing and returns a Figure."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m = Material("steel").add_load_case(150, 100, label="startup")
        fig = m.plot_goodman(show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
