"""Tests for the belt drive module."""

import math

import pytest

from mecapy import MechaElement
from mecapy.belts import Belt


class TestBelt:
    """Test cases for the Belt class."""

    def test_is_mecha_element(self):
        assert isinstance(Belt(), MechaElement)

    def test_wrap_angle_equal_pulleys(self):
        # Equal pulleys -> straight belt -> wrap angle of pi.
        assert Belt.wrap_angle(0.2, 0.2, 0.8) == pytest.approx(math.pi)

    def test_belt_length(self):
        D, d, C = 0.3, 0.15, 0.8
        expected = 2 * C + (math.pi / 2) * (D + d) + (D - d) ** 2 / (4 * C)
        assert Belt.belt_length(D, d, C) == pytest.approx(expected)

    def test_vbelt_ratio_exceeds_flat(self):
        theta = 2.9
        flat = Belt(belt_type="flat", friction=0.3)
        vbelt = Belt(belt_type="v", friction=0.3, groove_angle=38)
        assert vbelt.tension_ratio(theta) > flat.tension_ratio(theta)

    def test_flat_tension_ratio_formula(self):
        belt = Belt(belt_type="flat", friction=0.25)
        theta = 3.0
        assert belt.tension_ratio(theta) == pytest.approx(math.exp(0.25 * 3.0))

    def test_centrifugal_tension(self):
        belt = Belt(mass_per_length=0.4)
        assert belt.centrifugal_tension(20) == pytest.approx(0.4 * 400)

    def test_power(self):
        belt = Belt()
        assert belt.power(600, 200, 20) == pytest.approx(8000)

    def test_max_power_positive(self):
        belt = Belt(belt_type="v", friction=0.3, mass_per_length=0.4)
        theta = belt.wrap_angle(0.3, 0.15, 0.8)
        assert belt.max_power(600, 20, theta) > 0
