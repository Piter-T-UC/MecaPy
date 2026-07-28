"""Tests for the joints family: Key, Pin, Rivet."""

import math

import pytest

from mecapy.joints import Key, Pin, Rivet
from mecapy.materials import get_material_properties

SHEAR = 0.577
SY_STEEL = get_material_properties("steel")["yield_strength"] / 1e6


class TestKey:
    """Parallel key transmitting torque."""

    def test_force_and_stresses(self):
        """F = 2T/d; shear over w*L, bearing over (h/2)*L."""
        key = Key(width=8, height=8, length=40, shaft_diameter=30)
        T = 90e3  # N*mm
        f = 2 * T / 30
        assert key.tangential_force(T) == pytest.approx(f)
        assert key.shear_stress(T) == pytest.approx(f / (8 * 40))
        assert key.bearing_stress(T) == pytest.approx(f / ((8 / 2) * 40))

    def test_safety_factors(self):
        """Shear checked against 0.577*Sy, bearing against Sy."""
        key = Key(width=8, height=8, length=40, shaft_diameter=30)
        T = 90e3
        assert key.shear_safety_factor(T) == pytest.approx(
            SHEAR * SY_STEEL / key.shear_stress(T))
        assert key.bearing_safety_factor(T) == pytest.approx(
            SY_STEEL / key.bearing_stress(T))

    def test_torque_capacity_matches_governing_mode(self):
        """torque_capacity inverts the governing (weaker) mode."""
        key = Key(width=8, height=8, length=40, shaft_diameter=30)
        cap = key.torque_capacity(safety_factor=2.0)
        governing = min(key.shear_safety_factor(cap),
                        key.bearing_safety_factor(cap))
        assert governing == pytest.approx(2.0)

    @pytest.mark.parametrize("kwargs", [
        {"width": 0, "height": 8, "length": 40, "shaft_diameter": 30},
        {"width": 8, "height": -1, "length": 40, "shaft_diameter": 30},
        {"width": 8, "height": 8, "length": 0, "shaft_diameter": 30},
        {"width": 8, "height": 8, "length": 40, "shaft_diameter": 0},
    ])
    def test_invalid_inputs_raise(self, kwargs):
        """Non-positive dimensions raise."""
        with pytest.raises(ValueError):
            Key(**kwargs)

    def test_non_positive_torque_raises(self):
        """Torque must be strictly positive."""
        key = Key(width=8, height=8, length=40, shaft_diameter=30)
        with pytest.raises(ValueError):
            key.shear_stress(0)

    def test_bad_safety_factor_raises(self):
        """torque_capacity rejects a non-positive safety factor."""
        key = Key(width=8, height=8, length=40, shaft_diameter=30)
        with pytest.raises(ValueError):
            key.torque_capacity(0)

    def test_length_change_recomputes(self):
        """Doubling the length halves the shear stress."""
        key = Key(width=8, height=8, length=40, shaft_diameter=30)
        before = key.shear_stress(90e3)
        key.length = 80
        assert key.shear_stress(90e3) == pytest.approx(before / 2)

    def test_setter_revalidates(self):
        """A bad value after construction re-raises."""
        key = Key(width=8, height=8, length=40, shaft_diameter=30)
        with pytest.raises(ValueError):
            key.width = 0

    def test_repr(self):
        """repr names the class."""
        key = Key(width=8, height=8, length=40, shaft_diameter=30)
        assert key.__repr__().startswith("Key(")


class TestRivet:
    """Rivet in single/double shear and bearing."""

    def test_areas(self):
        """Double shear doubles the shear area; bearing is d*t."""
        r = Rivet(diameter=12, plate_thickness=8, shear_planes=2)
        assert r.shear_area == pytest.approx(2 * math.pi * 12 ** 2 / 4)
        assert r.bearing_area == pytest.approx(12 * 8)

    def test_stresses(self):
        """Shear and bearing stress from a transverse load."""
        r = Rivet(diameter=12, plate_thickness=8)
        F = 15000.0
        assert r.shear_stress(F) == pytest.approx(F / (math.pi * 12 ** 2 / 4))
        assert r.bearing_stress(F) == pytest.approx(F / (12 * 8))

    def test_safety_factors(self):
        """Shear against 0.577*Sy, bearing against Sy."""
        r = Rivet(diameter=12, plate_thickness=8)
        F = 15000.0
        assert r.shear_safety_factor(F) == pytest.approx(
            SHEAR * SY_STEEL / r.shear_stress(F))
        assert r.bearing_safety_factor(F) == pytest.approx(
            SY_STEEL / r.bearing_stress(F))

    def test_allowable_force_governing(self):
        """allowable_force inverts the governing mode."""
        r = Rivet(diameter=12, plate_thickness=8)
        allow = r.allowable_force(safety_factor=2.5)
        governing = min(r.shear_safety_factor(allow), r.bearing_safety_factor(allow))
        assert governing == pytest.approx(2.5)

    def test_invalid_inputs_raise(self):
        """Bad diameter, thickness or shear-plane count raise."""
        with pytest.raises(ValueError):
            Rivet(diameter=0, plate_thickness=8)
        with pytest.raises(ValueError):
            Rivet(diameter=12, plate_thickness=-1)
        with pytest.raises(ValueError):
            Rivet(diameter=12, plate_thickness=8, shear_planes=0)
        with pytest.raises(ValueError):
            Rivet(diameter=12, plate_thickness=8, shear_planes=1.5)

    def test_zero_force_raises(self):
        """A zero load has no safety factor."""
        r = Rivet(diameter=12, plate_thickness=8)
        with pytest.raises(ValueError):
            r.shear_safety_factor(0)
        with pytest.raises(ValueError):
            r.bearing_safety_factor(0)

    def test_bad_safety_factor_raises(self):
        """allowable_force rejects a non-positive safety factor."""
        r = Rivet(diameter=12, plate_thickness=8)
        with pytest.raises(ValueError):
            r.allowable_force(0)

    def test_repr(self):
        """repr names the class."""
        assert Rivet(diameter=12, plate_thickness=8).__repr__().startswith("Rivet(")


class TestPin:
    """Shear pin, single/double shear and torque transmission."""

    def test_double_shear_area(self):
        """Two shear planes double the area."""
        p = Pin(diameter=10, shear_planes=2)
        assert p.shear_area == pytest.approx(2 * math.pi * 10 ** 2 / 4)

    def test_shear_stress_and_sf(self):
        """Direct transverse shear stress and safety factor."""
        p = Pin(diameter=10)
        F = 8000.0
        assert p.shear_stress(F) == pytest.approx(F / (math.pi * 10 ** 2 / 4))
        assert p.shear_safety_factor(F) == pytest.approx(
            SHEAR * SY_STEEL / p.shear_stress(F))

    def test_torque_mode(self):
        """A cross pin reacts F = 2T/d_shaft in shear."""
        p = Pin(diameter=10, shear_planes=2)
        T, d = 60e3, 30.0
        expected = p.shear_stress(2 * T / d)
        assert p.torque_shear_stress(T, d) == pytest.approx(expected)
        assert p.torque_safety_factor(T, d) == pytest.approx(
            SHEAR * SY_STEEL / expected)

    def test_allowable_force(self):
        """allowable_force = 0.577*Sy*A / n."""
        p = Pin(diameter=10, shear_planes=2)
        allow = p.allowable_force(3.0)
        assert p.shear_safety_factor(allow) == pytest.approx(3.0)

    def test_invalid_inputs_raise(self):
        """Bad diameter or shear-plane count raise."""
        with pytest.raises(ValueError):
            Pin(diameter=0)
        with pytest.raises(ValueError):
            Pin(diameter=10, shear_planes=0)

    def test_zero_force_and_bad_sf_raise(self):
        """Zero load / bad safety factor / bad torque inputs raise."""
        p = Pin(diameter=10)
        with pytest.raises(ValueError):
            p.shear_safety_factor(0)
        with pytest.raises(ValueError):
            p.allowable_force(0)
        with pytest.raises(ValueError):
            p.torque_shear_stress(0, 30)
        with pytest.raises(ValueError):
            p.torque_shear_stress(60e3, 0)

    def test_repr(self):
        """repr names the class."""
        assert Pin(diameter=10).__repr__().startswith("Pin(")
