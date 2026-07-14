"""Tests for the Material class and database."""

import pytest

from mecapy import Material, get_material, get_available_materials
from mecapy.gears import Gear


class TestMaterial:
    """Test cases for the Material class."""

    def test_get_material_returns_material(self):
        steel = get_material("steel")
        assert isinstance(steel, Material)
        assert steel.name == "steel"
        assert steel.elastic_modulus == 210e9
        assert steel.ultimate_strength == 400e6

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError):
            get_material("unobtainium")

    def test_shear_modulus_derived_when_missing(self):
        mat = Material("m", elastic_modulus=210e9, poisson_ratio=0.3)
        assert mat.shear_modulus == pytest.approx(210e9 / (2 * 1.3))

    def test_dict_style_access(self):
        steel = get_material("steel")
        assert steel["yield_strength"] == 250e6
        assert "density" in steel
        assert steel.get("missing", 42) == 42

    def test_to_dict_has_all_fields(self):
        steel = get_material("steel")
        d = steel.to_dict()
        assert set(d) == set(Material.FIELDS)

    def test_available_materials(self):
        materials = get_available_materials()
        for name in ("steel", "aluminum", "copper", "cast_iron"):
            assert name in materials

    def test_material_instance_on_element(self):
        titanium = Material("titanium", elastic_modulus=114e9,
                            poisson_ratio=0.34, yield_strength=880e6)
        gear = Gear(teeth=18, module=2.0, material=titanium)
        assert gear.material_properties["elastic_modulus"] == 114e9
