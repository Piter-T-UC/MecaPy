"""Tests for bearing module."""

import pytest
from mecapy.bearings import Bearing


class TestBearing:
    """Test cases for Bearing class."""

    def test_bearing_creation(self):
        """Test creating a bearing object."""
        bearing = Bearing(
            bore_diameter=10.0,
            outer_diameter=26.0,
            width=8.0,
            bearing_type="ball",
        )
        assert bearing.bore_diameter == 10.0
        assert bearing.outer_diameter == 26.0
        assert bearing.width == 8.0
        assert bearing.bearing_type == "ball"

    def test_bearing_repr(self):
        """Test bearing string representation."""
        bearing = Bearing(
            bore_diameter=20.0,
            outer_diameter=52.0,
            width=15.0,
            bearing_type="roller",
        )
        assert "Bearing" in repr(bearing)
