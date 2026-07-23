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
        """Safety factor is yield strength over applied stress (ductile)."""
        element = MechaElement(material="steel")
        stress = 125e6  # half of steel's 250 MPa yield strength
        assert element.safety_factor(stress) == pytest.approx(2.0)

    def test_safety_factor_zero_stress(self):
        """Zero stress raises ValueError."""
        element = MechaElement()
        with pytest.raises(ValueError):
            element.safety_factor(0)

    def test_brittle_material_uses_ultimate_not_yield(self):
        """A brittle material (cast iron) rates against Sut, not Sy."""
        element = MechaElement(material="cast_iron")
        stress = 125e6
        # Coulomb-Mohr in tension reduces to Sut/sigma = 250/125 = 2.0,
        # which differs from the ductile Sy/sigma = 180/125 = 1.44.
        assert element.safety_factor(stress) == pytest.approx(250e6 / stress)
        assert element.safety_factor(stress) != pytest.approx(180e6 / stress)

    def test_brittle_material_compression_uses_suc(self):
        """In compression a brittle material rates against Suc."""
        element = MechaElement(material="cast_iron")
        assert element.safety_factor(-100e6) == pytest.approx(820e6 / 100e6)

    def test_brittle_flange_coupling_is_brittle_rated(self):
        """FlangeCoupling (default cast iron) inherits the brittle criterion."""
        from mecapy.couplings import FlangeCoupling

        coupling = FlangeCoupling(shaft_diameter=40.0, bolt_circle_diameter=120.0,
                                  n_bolts=4, bolt_diameter=12.0,
                                  flange_thickness=15.0)
        assert coupling.material == "cast_iron"
        assert coupling.safety_factor(125e6) == pytest.approx(250e6 / 125e6)


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

        bolt = Bolt(size="M10", length=50.0)
        # 5000 N over an 80 mm^2 stress area -> 62.5 MPa
        assert bolt.calculate_stress(5000, 80) == pytest.approx(62.5)
