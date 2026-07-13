"""Tests for wheel module."""

import pytest
from mecapy.wheels import Wheel


class TestWheel:
    """Test cases for Wheel class."""

    def test_wheel_creation(self):
        """Test creating a wheel object."""
        wheel = Wheel(radius=0.5, material="steel", mass=10.0)
        assert wheel.radius == 0.5
        assert wheel.material == "steel"
        assert wheel.mass == 10.0

    def test_wheel_repr(self):
        """Test wheel string representation."""
        wheel = Wheel(radius=0.25, material="aluminum", mass=5.0)
        assert "Wheel" in repr(wheel)
        assert "0.25m" in repr(wheel)
