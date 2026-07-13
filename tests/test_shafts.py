"""Tests for shaft module."""

import pytest
from mecapy.shafts import Shaft


class TestShaft:
    """Test cases for Shaft class."""

    def test_shaft_creation(self):
        """Test creating a shaft object."""
        shaft = Shaft(diameter=25.0, length=500.0, material="steel")
        assert shaft.diameter == 25.0
        assert shaft.length == 500.0
        assert shaft.material == "steel"

    def test_shaft_repr(self):
        """Test shaft string representation."""
        shaft = Shaft(diameter=30.0, length=1000.0, material="steel")
        assert "Shaft" in repr(shaft)
        assert "30.0mm" in repr(shaft)
