"""Tests for the roller chain module."""

import math

import pytest

from mecapy import MechaElement
from mecapy.chains import Chain


class TestChain:
    """Test cases for the Chain class."""

    def test_is_mecha_element(self):
        assert isinstance(Chain(pitch=12.7, teeth=17), MechaElement)

    def test_pitch_diameter(self):
        chain = Chain(pitch=12.7, teeth=17)
        expected = 12.7 / math.sin(math.pi / 17)
        assert chain.pitch_diameter() == pytest.approx(expected)

    def test_pitch_diameter_override_teeth(self):
        chain = Chain(pitch=12.7, teeth=17)
        expected = 12.7 / math.sin(math.pi / 40)
        assert chain.pitch_diameter(40) == pytest.approx(expected)

    def test_length_in_pitches(self):
        chain = Chain(pitch=12.7, teeth=17)
        N1, N2, C = 17, 40, 30
        expected = 2 * C + (N1 + N2) / 2 + (N2 - N1) ** 2 / (4 * math.pi ** 2 * C)
        assert chain.length_in_pitches(40, 30) == pytest.approx(expected)

    def test_velocity(self):
        chain = Chain(pitch=12.7, teeth=17)
        assert chain.velocity(1200) == pytest.approx(17 * 12.7 * 1200 / 60)

    def test_chordal_variation_positive(self):
        chain = Chain(pitch=12.7, teeth=17)
        assert chain.chordal_speed_variation() > 0
