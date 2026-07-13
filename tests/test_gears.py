"""Tests for gear module."""

import pytest
from mecapy.gears import Gear


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
