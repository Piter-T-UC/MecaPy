"""Tests for weld module."""

import pytest
from mecapy.welds import Weld


class TestWeld:
    """Test cases for Weld class."""

    def test_weld_creation(self):
        """Test creating a weld object."""
        weld = Weld(weld_type="fillet", material="steel", size=6.0)
        assert weld.weld_type == "fillet"
        assert weld.material == "steel"
        assert weld.size == 6.0

    def test_weld_repr(self):
        """Test weld string representation."""
        weld = Weld(weld_type="butt", material="steel", size=8.0)
        assert "Weld" in repr(weld)
        assert "fillet" in "fillet" or "butt" in repr(weld)
