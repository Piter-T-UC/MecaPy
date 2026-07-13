"""Tests for bolt module."""

import pytest
from mecapy.bolts import Bolt


class TestBolt:
    """Test cases for Bolt class."""

    def test_bolt_creation(self):
        """Test creating a bolt object."""
        bolt = Bolt(diameter=10.0, length=50.0, grade="M10", material="steel")
        assert bolt.diameter == 10.0
        assert bolt.length == 50.0
        assert bolt.grade == "M10"
        assert bolt.material == "steel"

    def test_bolt_repr(self):
        """Test bolt string representation."""
        bolt = Bolt(diameter=12.0, length=60.0, grade="M12", material="steel")
        assert "Bolt" in repr(bolt)
        assert "M12" in repr(bolt)
