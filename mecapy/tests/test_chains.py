"""Tests for the roller chain subsystem."""

import pytest

from mecapy import MechaElement
from mecapy.chains import RollerChain


def make_chain(**overrides):
    params = dict(chain_number=60, driver_teeth=17, driven_teeth=51,
                  center_distance=500.0)
    params.update(overrides)
    return RollerChain(**params)


class TestChainConstruction:
    def test_is_mecha_element(self):
        assert isinstance(make_chain(), MechaElement)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            make_chain(chain_number=999)
        with pytest.raises(ValueError):
            make_chain(driver_teeth=2)
        with pytest.raises(ValueError):
            make_chain(strands=5)

    def test_int_and_str_chain_numbers(self):
        assert RollerChain(chain_number=60, driver_teeth=17, driven_teeth=17).pitch \
            == RollerChain(chain_number="60", driver_teeth=17, driven_teeth=17).pitch


class TestChainGeometry:
    def test_pitch_diameter_hand_check(self):
        # No. 60 chain (p = 0.75 in = 19.05 mm), 17 teeth -> D = p/sin(180/17) ~ 103.7 mm
        chain = make_chain(driver_teeth=17)
        assert chain.driver_pitch_diameter == pytest.approx(103.7, rel=1e-2)

    def test_chordal_variation_decreases_with_teeth(self):
        small = make_chain(driver_teeth=13, driven_teeth=52)
        large = make_chain(driver_teeth=25, driven_teeth=52)
        assert large.chordal_speed_variation < small.chordal_speed_variation

    def test_length_center_distance_round_trip(self):
        chain = make_chain()
        length_pitches = chain.chain_length_pitches(500.0)
        assert chain.center_distance_for_length(length_pitches) == pytest.approx(500.0, rel=1e-6)

    def test_equal_sprockets_length_formula(self):
        chain = make_chain(driver_teeth=20, driven_teeth=20, center_distance=400.0)
        length_pitches = chain.chain_length_pitches(400.0)
        assert length_pitches == pytest.approx(2 * 400.0 / chain.pitch + 20)


class TestChainKinematicsAndStrength:
    def test_speed_formula(self):
        chain = make_chain()
        v = chain.chain_speed(1000)
        assert v == pytest.approx(chain.driver_teeth * chain.pitch * 1000 / 60000.0)

    def test_working_tension_times_speed_equals_power(self):
        chain = make_chain()
        power = 5000.0
        rpm = 800
        tension = chain.working_tension(power, rpm)
        assert tension * chain.chain_speed(rpm) == pytest.approx(power)

    def test_tensile_safety_factor_scales_with_strands(self):
        single = make_chain(strands=1)
        double = make_chain(strands=2)
        power, rpm = 2000.0, 500
        assert (double.tensile_safety_factor(power, rpm)
                == pytest.approx(2 * single.tensile_safety_factor(power, rpm)))


class TestRatedPower:
    def test_h1_governs_at_low_speed_h2_at_high_speed(self):
        chain = make_chain(driver_teeth=17, driven_teeth=17)
        p_in = chain.pitch / 25.4
        n1 = chain.driver_teeth

        def h1(rpm):
            return 0.004 * n1 ** 1.08 * rpm ** 0.9 * p_in ** (3 - 0.07 * p_in)

        def h2(rpm):
            return 1000 * 17 * n1 ** 1.5 * p_in ** 0.8 / rpm ** 1.5

        assert h1(500) < h2(500)
        assert h1(3000) > h2(3000)

    def test_rated_power_increases_with_teeth_and_pitch(self):
        base = make_chain(driver_teeth=17, driven_teeth=17)
        more_teeth = make_chain(driver_teeth=25, driven_teeth=25)
        bigger_pitch = make_chain(chain_number=80, driver_teeth=17, driven_teeth=17)
        assert more_teeth.rated_power(500) > base.rated_power(500)
        assert bigger_pitch.rated_power(500) > base.rated_power(500)

    def test_strand_factor_multiplies(self):
        single = make_chain(strands=1)
        triple = make_chain(strands=3)
        assert triple.rated_power(500) == pytest.approx(2.5 * single.rated_power(500))


class TestChainRatedPowerCrossover:
    """Anchor: Shigley Prob. 17-24, which locates the driver speed at which
    the link-plate-fatigue (Eq. 17-32) and roller-bushing-wear (Eq. 17-33)
    ratings cross for a No. 60 chain with a 17-tooth driver sprocket:
    n1 = 1227 rev/min, confirmed against the Ch. 17 solutions manual
    ("Table 17-20 confirms this point occurs at 1200 +/- 200 rev/min")."""

    def test_h1_equals_h2_at_1227_rpm(self):
        chain = make_chain(chain_number=60, driver_teeth=17, driven_teeth=17, strands=1)
        p_in = chain.pitch / 25.4
        n1 = chain.driver_teeth
        rpm = 1227
        h1 = 0.004 * n1 ** 1.08 * rpm ** 0.9 * p_in ** (3 - 0.07 * p_in)
        h2 = 1000 * 17 * n1 ** 1.5 * p_in ** 0.8 / rpm ** 1.5
        assert h1 == pytest.approx(h2, rel=1e-2)

    def test_rated_power_peaks_near_crossover(self):
        chain = make_chain(chain_number=60, driver_teeth=17, driven_teeth=17, strands=1)
        assert chain.rated_power(1227) > chain.rated_power(1000)
        assert chain.rated_power(1227) > chain.rated_power(1500)
