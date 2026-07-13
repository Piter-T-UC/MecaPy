"""Tests for beam module."""

import pytest
from mecapy.beams import Beam


class TestBeam:
    """Test cases for Beam class."""

    def test_beam_creation(self):
        """Test creating a beam object."""
        beam = Beam(length=5.0, material="steel")
        assert beam.length == 5.0
        assert beam.material == "steel"

    def test_beam_with_section(self):
        """Test beam creation with cross-section properties."""
        section = {"width": 0.1, "height": 0.2}
        beam = Beam(length=3.0, material="aluminum", section=section)
        assert beam.section == section

    def test_beam_repr(self):
        """Test beam string representation."""
        beam = Beam(length=5.0, material="steel")
        assert "Beam" in repr(beam)
        assert "5.0m" in repr(beam)
