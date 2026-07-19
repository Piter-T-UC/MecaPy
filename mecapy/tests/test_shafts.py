"""Tests for shaft module."""

import math

import pytest

from mecapy import MechaElement
from mecapy.shafts import Shaft


class TestShaft:
    """Test cases for Shaft class."""

    def test_shaft_creation(self):
        """Test creating a shaft object."""
        shaft = Shaft(diameter=25.0, length=500.0, material="steel")
        assert shaft.diameter == 25.0
        assert shaft.length == 500.0
        assert shaft.material == "steel"

    def test_shaft_is_mecha_element(self):
        """Shaft inherits from MechaElement."""
        shaft = Shaft(diameter=25.0, length=500.0)
        assert isinstance(shaft, MechaElement)

    def test_polar_moment(self):
        """Polar moment of area is pi * d^4 / 32."""
        shaft = Shaft(diameter=20.0, length=500.0)
        assert shaft.polar_moment == pytest.approx(math.pi * 20.0 ** 4 / 32)

    def test_torsional_stress(self):
        """Torsional shear stress is T*r/J."""
        shaft = Shaft(diameter=20.0, length=500.0)
        torque = 1e5  # N*mm
        expected = torque * 10.0 / (math.pi * 20.0 ** 4 / 32)
        assert shaft.torsional_stress(torque) == pytest.approx(expected)

    def test_shaft_repr(self):
        """Test shaft string representation."""
        shaft = Shaft(diameter=30.0, length=1000.0, material="steel")
        assert "Shaft" in repr(shaft)
        assert "30.0mm" in repr(shaft)
