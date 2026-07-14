"""Tests for the AGMA gear module."""

import pytest

from mecapy import MechaElement
from mecapy.gears import Gear, agma


class TestGear:
    """Test cases for the Gear class."""

    def test_gear_creation(self):
        gear = Gear(teeth=20, module=2.5, material="steel")
        assert gear.teeth == 20
        assert gear.module == 2.5
        assert gear.material == "steel"

    def test_is_mecha_element(self):
        assert isinstance(Gear(teeth=20, module=2.5), MechaElement)

    def test_pitch_diameter(self):
        assert Gear(teeth=20, module=2.5).pitch_diameter == 50.0

    def test_default_face_width(self):
        assert Gear(teeth=20, module=2.5).face_width == 25.0

    def test_pitch_line_velocity(self):
        import math
        gear = Gear(teeth=17, module=2.5)
        expected = math.pi * 42.5 * 1200 / 60000
        assert gear.pitch_line_velocity(1200) == pytest.approx(expected)

    def test_tangential_load(self):
        gear = Gear(teeth=17, module=2.5)
        v = gear.pitch_line_velocity(1200)
        assert gear.tangential_load(5000, 1200) == pytest.approx(5000 / v)

    def test_dynamic_factor_above_one(self):
        gear = Gear(teeth=17, module=2.5, quality_number=6)
        assert gear.dynamic_factor(1200) > 1.0

    def test_bending_stress_matches_formula(self):
        gear = Gear(teeth=17, module=2.5, face_width=38, quality_number=6)
        Wt = gear.tangential_load(5000, 1200)
        Kv = gear.dynamic_factor(1200)
        expected = Wt * Kv * (1 / (38 * 2.5)) * (1 / 0.34)
        assert gear.bending_stress(5000, 1200, geometry_factor=0.34) == pytest.approx(expected)

    def test_contact_stress_positive(self):
        gear = Gear(teeth=17, module=2.5, face_width=38)
        assert gear.contact_stress(5000, 1200, gear_ratio=3.0) > 0

    def test_bending_safety_factor(self):
        gear = Gear(teeth=17, module=2.5, face_width=38)
        stress = gear.bending_stress(5000, 1200, geometry_factor=0.34)
        # steel endurance limit is 200 MPa
        assert gear.bending_safety_factor(stress) == pytest.approx(200 / stress)

    def test_gear_repr(self):
        assert "Gear" in repr(Gear(teeth=20, module=2.5))


class TestAgmaHelpers:
    """Test cases for AGMA helper functions."""

    def test_elastic_coefficient_steel(self):
        ze = agma.elastic_coefficient(210e9, 0.3, 210e9, 0.3)
        assert ze == pytest.approx(191.6, rel=1e-2)

    def test_pitting_geometry_factor_positive(self):
        assert agma.pitting_geometry_factor(20, 3.0) > 0

    def test_dynamic_factor_perfect_gear(self):
        # Higher quality number -> Kv closer to 1.
        low = agma.dynamic_factor(6, 10)
        high = agma.dynamic_factor(11, 10)
        assert high < low
