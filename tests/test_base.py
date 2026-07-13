"""Tests for the MechaElement base class and inheritance."""

import pytest

from mecapy import MechaElement
from mecapy.beams import Beam
from mecapy.gears import Gear
from mecapy.shafts import Shaft


class TestMechaElement:
    """Test cases for the MechaElement base class."""

    def test_material_properties(self):
        """Base class exposes material properties from the database."""
        element = MechaElement(material="steel")
        props = element.material_properties
        assert props["elastic_modulus"] == 210e9

    def test_calculate_stress(self):
        """Axial stress is force divided by area."""
        element = MechaElement(material="steel")
        assert element.calculate_stress(1000, 0.01) == pytest.approx(1e5)

    def test_calculate_stress_invalid_area(self):
        """A non-positive area raises ValueError."""
        element = MechaElement()
        with pytest.raises(ValueError):
            element.calculate_stress(1000, 0)

    def test_safety_factor(self):
        """Safety factor is yield strength over applied stress."""
        element = MechaElement(material="steel")
        stress = 125e6  # half of steel's 250 MPa yield strength
        assert element.safety_factor(stress) == pytest.approx(2.0)

    def test_safety_factor_zero_stress(self):
        """Zero stress raises ValueError."""
        element = MechaElement()
        with pytest.raises(ValueError):
            element.safety_factor(0)


class TestInheritance:
    """Every element should inherit from MechaElement."""

    @pytest.mark.parametrize(
        "element",
        [
            Beam(length=5.0),
            Gear(teeth=20, module=2.5),
            Shaft(diameter=25.0, length=500.0),
        ],
    )
    def test_is_mecha_element(self, element):
        """Elements are instances of the base class."""
        assert isinstance(element, MechaElement)

    def test_inherited_stress_on_bolt(self):
        """Elements inherit calculate_stress from the base class."""
        from mecapy.bolts import Bolt

        bolt = Bolt(diameter=10.0, length=50.0)
        # 5000 N over an 80 mm^2 stress area -> 62.5 MPa
        assert bolt.calculate_stress(5000, 80) == pytest.approx(62.5)
