"""Tests for the Column (buckling) element."""

import math

import pytest

from mecapy.columns import Column


class TestColumnConstruction:
    """Construction, validation and settable inputs."""

    def test_basic_geometry(self):
        """Radius of gyration and slenderness follow k = sqrt(I/A)."""
        col = Column.circular(diameter=20, length=1000)
        assert col.radius_of_gyration == pytest.approx(20 / 4)  # d/4
        assert col.effective_length == pytest.approx(1000)
        assert col.slenderness_ratio == pytest.approx(1000 / 5)

    def test_end_condition_scales_effective_length(self):
        """Fixed-fixed (K=0.5) halves the effective length."""
        col = Column.circular(diameter=20, length=1000, end_condition=0.5)
        assert col.effective_length == pytest.approx(500)

    @pytest.mark.parametrize("kwargs", [
        {"length": 0, "area": 10, "second_moment": 10},
        {"length": 100, "area": -1, "second_moment": 10},
        {"length": 100, "area": 10, "second_moment": 0},
        {"length": 100, "area": 10, "second_moment": 10, "end_condition": 0},
    ])
    def test_invalid_inputs_raise(self, kwargs):
        """Non-positive dimensions or end factor raise ValueError."""
        with pytest.raises(ValueError):
            Column(**kwargs)

    def test_circular_rejects_bad_diameter(self):
        """circular() rejects a non-positive diameter."""
        with pytest.raises(ValueError):
            Column.circular(diameter=0, length=100)

    def test_rectangular_uses_weak_axis(self):
        """rectangular() buckles about the weak (smaller-I) axis."""
        col = Column.rectangular(width=30, depth=10, length=500)
        assert col.area == pytest.approx(300)
        # weak axis: I = max_side * min_side^3 / 12 = 30*10^3/12
        assert col.second_moment == pytest.approx(30 * 10 ** 3 / 12)

    def test_rectangular_rejects_bad_dims(self):
        """rectangular() rejects non-positive sides."""
        with pytest.raises(ValueError):
            Column.rectangular(width=0, depth=10, length=100)


class TestColumnBuckling:
    """Euler / Johnson regime selection and safety factors."""

    def test_slender_column_uses_euler(self):
        """A long thin column is slender; critical load is the Euler load."""
        col = Column.circular(diameter=20, length=1500)
        assert col.is_slender
        assert col.critical_load == pytest.approx(col.euler_load)

    def test_euler_formula(self):
        """Euler load matches pi^2 E I / Le^2."""
        col = Column.circular(diameter=20, length=1500)
        E = col.elastic_modulus
        expected = math.pi ** 2 * E * col.second_moment / col.effective_length ** 2
        assert col.euler_load == pytest.approx(expected)

    def test_short_column_uses_johnson(self):
        """A stocky column is not slender; critical load is the Johnson load."""
        col = Column.circular(diameter=50, length=250)
        assert not col.is_slender
        assert col.critical_load == pytest.approx(col.johnson_load)

    def test_johnson_matches_euler_at_transition(self):
        """The two curves are tangent: equal load at the transition slenderness."""
        # Tune length so slenderness == transition slenderness.
        col = Column.circular(diameter=30, length=1000)
        target_len = col.transition_slenderness * col.radius_of_gyration
        col.length = target_len
        assert col.slenderness_ratio == pytest.approx(col.transition_slenderness)
        assert col.euler_load == pytest.approx(col.johnson_load, rel=1e-9)

    def test_critical_stress(self):
        """Critical stress is the critical load over the area."""
        col = Column.circular(diameter=20, length=1500)
        assert col.critical_stress == pytest.approx(col.critical_load / col.area)

    def test_buckling_safety_factor(self):
        """Buckling SF is the critical load over the applied load."""
        col = Column.circular(diameter=20, length=1500)
        load = col.critical_load / 3
        assert col.buckling_safety_factor(load) == pytest.approx(3.0)

    def test_euler_safety_factor(self):
        """Euler SF uses the Euler load specifically."""
        col = Column.circular(diameter=20, length=1500)
        assert col.euler_safety_factor(col.euler_load / 2) == pytest.approx(2.0)

    def test_safety_factor_requires_positive_load(self):
        """A non-positive load raises."""
        col = Column.circular(diameter=20, length=1500)
        with pytest.raises(ValueError):
            col.buckling_safety_factor(0)
        with pytest.raises(ValueError):
            col.euler_safety_factor(-10)


class TestColumnRecompute:
    """Recompute-on-change guarantee for the settable inputs."""

    def test_length_change_updates_slenderness(self):
        """Changing the length updates the derived slenderness immediately."""
        col = Column.circular(diameter=20, length=1000)
        before = col.slenderness_ratio
        col.length = 2000
        assert col.slenderness_ratio == pytest.approx(2 * before)

    def test_setter_revalidates(self):
        """Assigning a bad value after construction re-raises."""
        col = Column.circular(diameter=20, length=1000)
        with pytest.raises(ValueError):
            col.area = -5
        with pytest.raises(ValueError):
            col.end_condition = 0


class TestSecantFormula:
    """Eccentric loading via the secant formula."""

    def test_secant_exceeds_direct_stress(self):
        """Eccentricity raises the peak stress above the direct P/A value."""
        col = Column.circular(diameter=30, length=1000)
        load = 5000.0
        direct = load / col.area
        peak = col.secant_max_stress(load, eccentricity=2.0,
                                     extreme_fiber_distance=15.0)
        assert peak > direct

    def test_zero_eccentricity_recovers_direct_stress(self):
        """With no eccentricity the secant stress is just P/A."""
        col = Column.circular(diameter=30, length=1000)
        load = 5000.0
        assert col.secant_max_stress(load, 0.0, 15.0) == pytest.approx(
            load / col.area)

    def test_secant_validation(self):
        """Non-positive load or negative geometry raises."""
        col = Column.circular(diameter=30, length=1000)
        with pytest.raises(ValueError):
            col.secant_max_stress(0, 2.0, 15.0)
        with pytest.raises(ValueError):
            col.secant_max_stress(5000, -1.0, 15.0)

    def test_repr(self):
        """repr names the class and key inputs."""
        col = Column.circular(diameter=20, length=1000)
        assert col.__repr__().startswith("Column(")
