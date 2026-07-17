"""Tests for bolt module."""

from math import isclose, sqrt

import pytest

from mecapy.bolts import Bolt, BoltedUnion


class TestBolt:
    """Test cases for Bolt class."""

    def test_bolt_creation(self):
        """Test creating a bolt object."""
        bolt = Bolt(size="M10", length=50.0, property_class="8.8", material="steel")
        assert bolt.size == "M10"
        assert bolt.length == 50.0
        assert bolt.property_class == "8.8"
        assert bolt.material == "steel"

    def test_bolt_repr(self):
        """Test bolt string representation."""
        bolt = Bolt(size="M12", length=60.0)
        assert "Bolt" in repr(bolt)
        assert "M12" in repr(bolt)

    def test_thread_geometry(self):
        """Test pitch, diameters and areas from the ISO table."""
        bolt = Bolt(size="M10", length=50.0)
        assert bolt.pitch == 1.5
        assert bolt.nominal_diameter == 10.0
        assert bolt.diameter == 10.0
        assert bolt.stress_area == 58.0
        assert isclose(bolt.nominal_area, 78.5398, rel_tol=1e-4)

    def test_strength_values(self):
        """Test property-class strengths and proof load."""
        bolt = Bolt(size="M10", length=50.0, property_class="8.8")
        assert bolt.proof_strength == 580.0
        assert bolt.yield_strength == 640.0
        assert bolt.tensile_strength == 800.0
        assert isclose(bolt.proof_load, 580.0 * 58.0)
        assert isclose(bolt.recommended_preload, 0.75 * 580.0 * 58.0)

    def test_stiffness_and_elongation(self):
        """Test the axial stiffness model and elongation round-trip."""
        bolt = Bolt(size="M10", length=50.0, material="steel")
        # k = As * E / L with E = 210000 MPa for steel
        expected_k = 58.0 * 210000.0 / 50.0
        assert isclose(bolt.stiffness, expected_k)
        force = 10000.0
        assert isclose(bolt.elongation(force), force / expected_k)
        # round-trip: k * delta == F
        assert isclose(bolt.stiffness * bolt.elongation(force), force)

    def test_tensile_stress_and_safety_factor(self):
        """Test tensile stress on the stress area and safety factor."""
        bolt = Bolt(size="M10", length=50.0, property_class="8.8")
        assert isclose(bolt.tensile_stress(5800.0), 100.0)
        assert isclose(bolt.bolt_safety_factor(5800.0), 640.0 / 100.0)

    def test_invalid_inputs(self):
        """Test that non-physical or unknown inputs raise ValueError."""
        with pytest.raises(ValueError):
            Bolt(size="M11", length=50.0)
        with pytest.raises(ValueError):
            Bolt(size="M10", length=50.0, property_class="9.9")
        with pytest.raises(ValueError):
            Bolt(size="M10", length=0.0)
        with pytest.raises(ValueError):
            Bolt(size="M10", length=50.0).bolt_safety_factor(0.0)


def square_pattern():
    """4-bolt square pattern, 100 mm side, centered at (50, 50)."""
    return [[1, 0.0, 0.0], [2, 100.0, 0.0], [3, 100.0, 100.0], [4, 0.0, 100.0]]


class TestBoltedUnion:
    """Test cases for BoltedUnion class."""

    def test_centroid(self):
        """Test centroid of a symmetric square pattern."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern())
        assert union.centroid == (50.0, 50.0)
        assert union.n_bolts == 4

    def test_direct_shear(self):
        """Pure in-plane force splits equally among the bolts."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(), forces=(4000.0, 0.0, 0.0))
        for entry in union.bolt_forces().values():
            assert isclose(entry["shear"][0], 1000.0)
            assert isclose(entry["shear"][1], 0.0, abs_tol=1e-12)
            assert isclose(entry["shear_magnitude"], 1000.0)
            assert entry["axial"] == 0.0

    def test_torsion(self):
        """Pure torsion gives tangential shear Mz*r/sum(r^2), zero resultant."""
        mz = 1e6
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(), moments=(0.0, 0.0, mz))
        forces = union.bolt_forces()
        r = sqrt(2) * 50.0
        sum_r2 = 4 * r ** 2
        expected = mz * r / sum_r2
        total_x = total_y = 0.0
        for number, entry in forces.items():
            assert isclose(entry["shear_magnitude"], expected)
            total_x += entry["shear"][0]
            total_y += entry["shear"][1]
        assert isclose(total_x, 0.0, abs_tol=1e-9)
        assert isclose(total_y, 0.0, abs_tol=1e-9)
        # bolt 1 at (-50, -50) relative: force (-Mz*dy, Mz*dx)/sum_r2 = (+, -)
        fsx, fsy = forces[1]["shear"]
        assert fsx > 0 and fsy < 0

    def test_shear_component_breakdown(self):
        """shear_direct + shear_torsion sum to shear, with correct signs."""
        mz = 1e6
        union = BoltedUnion(
            Bolt("M10", 50.0), square_pattern(),
            forces=(4000.0, 0.0, 0.0), moments=(0.0, 0.0, mz),
        )
        forces = union.bolt_forces()
        sum_r2 = 4 * (sqrt(2) * 50.0) ** 2
        for number, entry in forces.items():
            vx, vy = entry["shear_direct"]
            tx, ty = entry["shear_torsion"]
            assert isclose(vx, 1000.0)
            assert isclose(vy, 0.0, abs_tol=1e-12)
            assert isclose(vx + tx, entry["shear"][0])
            assert isclose(vy + ty, entry["shear"][1])
        # Bolt 1 at (-50, -50) relative: (tx, ty) = (-Mz*dy, Mz*dx)/sum_r2
        tx, ty = forces[1]["shear_torsion"]
        assert isclose(tx, mz * 50.0 / sum_r2)
        assert isclose(ty, -mz * 50.0 / sum_r2)

    def test_shear_torsion_zero_without_mz(self):
        """No torsion moment -> torsion components are exactly (0, 0)."""
        union = BoltedUnion(
            Bolt("M10", 50.0), square_pattern(), forces=(4000.0, -2000.0, 0.0)
        )
        for entry in union.bolt_forces().values():
            assert entry["shear_torsion"] == (0.0, 0.0)
            assert entry["shear"] == entry["shear_direct"]

    def test_direct_axial(self):
        """Pure axial force splits equally."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(), forces=(0.0, 0.0, 8000.0))
        for entry in union.bolt_forces().values():
            assert isclose(entry["axial"], 2000.0)
            assert entry["shear_magnitude"] == 0.0

    def test_bending(self):
        """Pure Mx bending loads bolts proportionally to their y offset."""
        mx = 2e5
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(), moments=(mx, 0.0, 0.0))
        forces = union.bolt_forces()
        # dy = +/-50, sum_dy2 = 4*2500 = 10000 -> axial = +/- mx*50/10000
        expected = mx * 50.0 / 10000.0
        assert isclose(forces[3]["axial"], expected)   # y = 100 (dy = +50), tension
        assert isclose(forces[4]["axial"], expected)
        assert isclose(forces[1]["axial"], -expected)  # y = 0 (dy = -50), compression
        assert isclose(forces[2]["axial"], -expected)

    def test_equilibrium(self):
        """Per-bolt forces re-sum to the applied loads under combined loading."""
        union = BoltedUnion(
            Bolt("M12", 60.0), square_pattern(),
            forces=(3000.0, -2000.0, 5000.0), moments=(1e5, -2e5, 5e5),
        )
        forces = union.bolt_forces()
        x_bar, y_bar = union.centroid
        sum_fx = sum(e["shear"][0] for e in forces.values())
        sum_fy = sum(e["shear"][1] for e in forces.values())
        sum_fz = sum(e["axial"] for e in forces.values())
        sum_mx = sum_my = sum_mz = 0.0
        for row in union.positions:
            number, x, y = row
            dx, dy = x - x_bar, y - y_bar
            fsx, fsy = forces[number]["shear"]
            fz = forces[number]["axial"]
            sum_mz += dx * fsy - dy * fsx
            sum_mx += dy * fz
            sum_my += -dx * fz
        assert isclose(sum_fx, 3000.0)
        assert isclose(sum_fy, -2000.0)
        assert isclose(sum_fz, 5000.0)
        assert isclose(sum_mx, 1e5)
        assert isclose(sum_my, -2e5)
        assert isclose(sum_mz, 5e5)

    def test_max_loaded_bolt_and_safety_factors(self):
        """Most loaded bolt and per-bolt safety factors are consistent."""
        union = BoltedUnion(
            Bolt("M12", 60.0), square_pattern(),
            forces=(3000.0, 0.0, 5000.0), moments=(1e5, 0.0, 5e5),
        )
        number, entry = union.max_loaded_bolt()
        factors = union.safety_factors()
        assert set(factors) == {1, 2, 3, 4}
        assert factors[number] == min(factors.values())
        assert all(sf > 0 for sf in factors.values())

    def test_invalid_inputs(self):
        """Test constructor and distribution validation."""
        bolt = Bolt("M10", 50.0)
        with pytest.raises(ValueError):
            BoltedUnion("not a bolt", square_pattern())
        with pytest.raises(ValueError):
            BoltedUnion(bolt, [[1, 0.0]])
        with pytest.raises(ValueError):
            BoltedUnion(bolt, [[1, 0.0, 0.0], [1, 10.0, 0.0]])
        with pytest.raises(ValueError):
            BoltedUnion(bolt, [])
        with pytest.raises(ValueError):
            BoltedUnion(bolt, square_pattern(), forces=(1.0, 2.0))
        with pytest.raises(ValueError):
            BoltedUnion(bolt, square_pattern(), moments=(1.0,))
        # torsion with a single bolt at the centroid has no lever arm
        with pytest.raises(ValueError):
            BoltedUnion(bolt, [[1, 0.0, 0.0]], moments=(0.0, 0.0, 1e5)).bolt_forces()
        # bending Mx with all bolts on the x axis
        with pytest.raises(ValueError):
            BoltedUnion(
                bolt, [[1, 0.0, 0.0], [2, 100.0, 0.0]], moments=(1e5, 0.0, 0.0)
            ).bolt_forces()

    def test_plot_distribution_smoke(self):
        """Plot returns a matplotlib Figure without showing it."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        union = BoltedUnion(
            Bolt("M10", 50.0), square_pattern(),
            forces=(3000.0, 1000.0, 5000.0), moments=(1e5, 0.0, 5e5),
        )
        fig = union.plot_distribution(show=False)
        assert isinstance(fig, Figure)
