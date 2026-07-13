"""Tests for the SymPy-backed beam module."""

import pytest
from sympy import symbols

from mecapy import MechaElement
from mecapy.beams import Beam
from mecapy.beams import calculations


class TestBeam:
    """Test cases for the Beam class."""

    def test_beam_creation(self):
        """Test creating a beam object."""
        beam = Beam(length=5.0, material="steel")
        assert beam.length == 5.0
        assert beam.material == "steel"

    def test_beam_is_mecha_element(self):
        """Beam inherits from MechaElement."""
        beam = Beam(length=5.0)
        assert isinstance(beam, MechaElement)

    def test_default_elastic_modulus_from_material(self):
        """Elastic modulus defaults to the material database value."""
        beam = Beam(length=5.0, material="steel")
        assert beam.elastic_modulus == 210e9

    def test_simply_supported_reactions(self):
        """A central point load splits evenly between two supports."""
        E, I = symbols("E I", positive=True)
        beam = Beam(length=10, elastic_modulus=E, second_moment=I)
        beam.add_support(0, "pin").add_support(10, "roller")
        beam.add_point_load(-1000, 5)
        reactions = beam.reactions
        assert reactions[symbols("R_0")] == 500
        assert reactions[symbols("R_10")] == 500

    def test_simply_supported_max_moment(self):
        """Max bending moment of a central load is P*L/4 at midspan."""
        E, I = symbols("E I", positive=True)
        beam = Beam(length=10, elastic_modulus=E, second_moment=I)
        beam.add_support(0, "pin").add_support(10, "roller")
        beam.add_point_load(-1000, 5)
        location, moment = beam.max_bending_moment()
        assert location == 5
        assert moment == 2500  # 1000 * 10 / 4

    def test_cantilever_reactions(self):
        """A cantilever carries the full load and reacting moment."""
        E, I = symbols("E I", positive=True)
        beam = Beam(length=10, elastic_modulus=E, second_moment=I)
        beam.add_support(0, "fixed")
        beam.add_point_load(-1000, 10)
        reactions = beam.reactions
        assert reactions[symbols("R_0")] == 1000
        assert reactions[symbols("M_0")] == -10000

    def test_bending_stress(self):
        """Bending stress uses M*c/I from the max moment."""
        E = 210e9
        I = 8.0e-6
        beam = Beam(length=10, elastic_modulus=E, second_moment=I)
        beam.add_support(0, "pin").add_support(10, "roller")
        beam.add_point_load(-1000, 5)
        stress = beam.bending_stress(distance_to_fiber=0.1)
        assert stress == pytest.approx(2500 * 0.1 / I)

    def test_beam_repr(self):
        """Test beam string representation."""
        beam = Beam(length=5.0, material="steel")
        assert "Beam" in repr(beam)
        assert "5.0" in repr(beam)


class TestBeamCalculations:
    """Test cases for the standalone calculation helpers."""

    def test_cantilever_deflection(self):
        """Cantilever end deflection is P*L^3/(3*E*I)."""
        result = calculations.calculate_deflection(1000, 2.0, 210e9, 8e-6)
        assert result == pytest.approx(1000 * 2.0 ** 3 / (3 * 210e9 * 8e-6))

    def test_bending_stress_helper(self):
        """Bending stress helper computes M*c/I."""
        assert calculations.calculate_bending_stress(2500, 8e-6, 0.1) == pytest.approx(
            2500 * 0.1 / 8e-6
        )

    def test_shear_stress_helper(self):
        """Shear stress helper computes V/A."""
        assert calculations.calculate_shear_stress(1000, 0.01) == pytest.approx(1e5)
