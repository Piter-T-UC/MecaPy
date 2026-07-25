"""Tests for bolt module."""

from math import isclose, log, pi, radians, sqrt, tan

import pytest

from mecapy.bolts import Bolt, BoltedUnion, get_pitch, shigley_thread_geometry


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


class TestThreadDataHelpers:
    """Test cases for the pitch lookup and Shigley geometry helpers."""

    def test_get_pitch_by_size_and_diameter(self):
        """Pitch can be queried by designation or nominal diameter."""
        assert get_pitch("M10") == 1.5
        assert get_pitch(10) == 1.5
        assert get_pitch(10.0) == 1.5
        assert get_pitch("M24") == 3.0

    def test_get_pitch_unknown(self):
        """Unknown sizes and diameters raise ValueError."""
        with pytest.raises(ValueError):
            get_pitch("M11")
        with pytest.raises(ValueError):
            get_pitch(11.0)

    def test_shigley_geometry_m10(self):
        """Shigley formulas reproduce the ISO M10 stress area closely."""
        geometry = shigley_thread_geometry(10.0, 1.5)
        assert isclose(geometry["minor_diameter"], 10.0 - 1.226869 * 1.5)
        assert isclose(geometry["pitch_diameter"], 10.0 - 0.649519 * 1.5)
        # At = (pi/4)*((dp + dr)/2)^2 = 58.0 mm^2 within 0.1%
        assert isclose(geometry["stress_area"], 58.0, rel_tol=1e-3)

    def test_shigley_geometry_validation(self):
        """Non-physical diameter/pitch combinations raise ValueError."""
        with pytest.raises(ValueError):
            shigley_thread_geometry(0.0, 1.5)
        with pytest.raises(ValueError):
            shigley_thread_geometry(10.0, 0.0)
        with pytest.raises(ValueError):
            shigley_thread_geometry(1.0, 1.0)  # minor diameter <= 0


class TestCustomBolt:
    """Test cases for custom (diameter + pitch) bolts and the Shigley flag."""

    def test_custom_bolt_shigley_area(self):
        """Custom M10x1.5 matches the ISO table area within 0.1%."""
        bolt = Bolt(length=50.0, diameter=10.0, pitch=1.5)
        assert bolt.stress_area_method == "shigley"
        assert bolt.size == "M10x1.5"
        assert "M10x1.5" in repr(bolt)
        assert isclose(bolt.minor_diameter, 8.1596965)
        assert isclose(bolt.pitch_diameter, 9.0257215)
        assert isclose(bolt.stress_area, 58.0, rel_tol=1e-3)

    def test_custom_fine_pitch(self):
        """Fine-series M12x1.25 matches Shigley Table 8-1 (92.1 mm^2)."""
        bolt = Bolt(length=60.0, diameter=12.0, pitch=1.25)
        assert isclose(bolt.stress_area, 92.1, rel_tol=1e-3)
        assert bolt.pitch == 1.25

    def test_shigley_optin_on_table_size(self):
        """Table sizes keep the table area unless Shigley is requested."""
        table_bolt = Bolt("M10", 50.0)
        shigley_bolt = Bolt("M10", 50.0, stress_area_method="shigley")
        assert table_bolt.stress_area == 58.0
        assert table_bolt.stress_area_method == "table"
        assert shigley_bolt.stress_area_method == "shigley"
        assert isclose(shigley_bolt.stress_area, 58.0, rel_tol=1e-3)
        assert shigley_bolt.stress_area != 58.0

    def test_minor_pitch_diameter_on_table_bolt(self):
        """Table bolts also expose the Shigley minor/pitch diameters."""
        bolt = Bolt("M12", 60.0)
        assert isclose(bolt.minor_diameter, 12.0 - 1.226869 * 1.75)
        assert isclose(bolt.pitch_diameter, 12.0 - 0.649519 * 1.75)

    def test_custom_bolt_validation(self):
        """Inconsistent size/diameter/pitch combinations raise ValueError."""
        with pytest.raises(ValueError):
            Bolt(size="M10", length=50.0, diameter=10.0, pitch=1.5)
        with pytest.raises(ValueError):
            Bolt(length=50.0, pitch=1.5)
        with pytest.raises(ValueError):
            Bolt(length=50.0)
        with pytest.raises(ValueError):
            Bolt(length=50.0, diameter=1.0, pitch=1.0)
        with pytest.raises(ValueError):
            Bolt(length=50.0, diameter=10.0, pitch=1.5, stress_area_method="table")
        with pytest.raises(ValueError):
            Bolt("M10", 50.0, stress_area_method="bogus")
        with pytest.raises(ValueError):
            Bolt(diameter=10.0, pitch=1.5)  # missing length


def steel_plates():
    """Two 20 mm steel plates (grip 40 mm)."""
    return [(20.0, "steel"), (20.0, "steel")]


class TestBoltedUnionJoint:
    """Test cases for set_bolt, plates, member stiffness and preload."""

    def test_set_bolt(self):
        """set_bolt swaps the bolt and keeps the rest of the union."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(0.0, 0.0, 8000.0))
        union.set_bolt(Bolt("M12", 50.0))
        assert union.bolt.size == "M12"
        assert union.material == "steel"
        assert union.n_bolts == 4
        with pytest.raises(ValueError):
            union.set_bolt("not a bolt")

    def test_set_bolt_grip_check(self):
        """Bolts shorter than the grip are rejected."""
        with pytest.raises(ValueError):
            BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                        plates=[(30.0, "steel"), (30.0, "steel")])
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            plates=steel_plates())
        with pytest.raises(ValueError):
            union.set_bolt(Bolt("M10", 30.0))

    def test_member_stiffness_symmetric(self):
        """Two identical steel plates: hand-checked frustum stiffness.

        d = 10, dw = 15, each half is one frustum with t = 20, D = 15:
        ln argument = (2*tan(30)*20+5)*25 / ((2*tan(30)*20+25)*5),
        k1 = tan(30)*pi*210000*10 / ln(ln_arg) ~= 3.5537e6 N/mm,
        two in series -> km ~= 1.7769e6 N/mm.
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            plates=steel_plates())
        assert union.grip == 40.0
        tan30 = tan(radians(30))
        ln_arg = (2 * tan30 * 20.0 + 5.0) * 25.0 / ((2 * tan30 * 20.0 + 25.0) * 5.0)
        expected_k1 = tan30 * pi * 210000.0 * 10.0 / log(ln_arg)
        assert isclose(union.member_stiffness, expected_k1 / 2, rel_tol=1e-9)
        assert isclose(union.member_stiffness, 1.7769e6, rel_tol=1e-3)
        # kb = As*E/grip = 58*210000/40
        assert isclose(union.bolt_stiffness, 304500.0)
        expected_c = 304500.0 / (304500.0 + expected_k1 / 2)
        assert isclose(union.joint_constant, expected_c, rel_tol=1e-9)
        assert isclose(union.joint_constant, 0.1463, rel_tol=1e-3)

    def test_member_stiffness_asymmetric(self):
        """Mixed stack split at the midplane: 3 frusta recomputed longhand.

        Plates: 10 mm steel + 30 mm aluminum, grip 40, midplane at 20.
        Top half: steel t=10 D=15, aluminum t=10 D=15+2*tan(30)*10.
        Bottom half: aluminum t=20 D=15.
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            plates=[(10.0, "steel"), (30.0, "aluminum")])
        e_steel, e_alu = 210000.0, 69000.0  # database values in MPa
        d, dw = 10.0, 15.0
        tan30 = tan(radians(30))

        def frustum_k(t, big_d, modulus):
            ln_arg = ((2 * tan30 * t + big_d - d) * (big_d + d)) / (
                (2 * tan30 * t + big_d + d) * (big_d - d))
            return tan30 * pi * modulus * d / log(ln_arg)

        k1 = frustum_k(10.0, dw, e_steel)
        k2 = frustum_k(10.0, dw + 2 * tan30 * 10.0, e_alu)
        k3 = frustum_k(20.0, dw, e_alu)
        expected_km = 1 / (1 / k1 + 1 / k2 + 1 / k3)
        assert isclose(union.member_stiffness, expected_km, rel_tol=1e-9)
        # softer than the all-steel symmetric stack of the same grip
        all_steel = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                                plates=steel_plates())
        assert union.member_stiffness < all_steel.member_stiffness

    def test_bolt_tensions_separation_proof(self):
        """Preload-aware distribution and factors, hand-checked chain.

        M10 8.8: Fi = 0.75*580*58 = 25230 N; Fz = 40000 -> P = 10000 N.
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(0.0, 0.0, 40000.0), plates=steel_plates())
        c = union.joint_constant
        fi = 25230.0
        assert isclose(union.effective_preload, fi)
        tensions = union.bolt_tensions()
        for entry in tensions.values():
            assert isclose(entry["external"], 10000.0)
            assert isclose(entry["bolt_tension"], fi + c * 10000.0)
            assert isclose(entry["member_force"], (1 - c) * 10000.0 - fi)
            assert entry["member_force"] < 0  # still clamped
        for n0 in union.separation_safety_factors().values():
            assert isclose(n0, fi / ((1 - c) * 10000.0))
            assert isclose(n0, 2.955, rel_tol=1e-3)
        for nl in union.proof_safety_factors().values():
            assert isclose(nl, (580.0 * 58.0 - fi) / (c * 10000.0))
            assert isclose(nl, 5.748, rel_tol=1e-3)

    def test_slip_safety_factors(self):
        """Slip factor is friction capacity over bolt shear."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 0.0), plates=steel_plates())
        # V = 2000 N/bolt, P = 0 -> clamp = Fi = 25230 N
        for n_slip in union.slip_safety_factors(mu=0.2).values():
            assert isclose(n_slip, 0.2 * 25230.0 / 2000.0)
        with pytest.raises(ValueError):
            union.slip_safety_factors(mu=0.0)

    def test_explicit_preload(self):
        """Explicit preload overrides the recommended value everywhere."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(0.0, 0.0, 40000.0), plates=steel_plates(),
                            preload=20000.0)
        c = union.joint_constant
        assert union.effective_preload == 20000.0
        for n0 in union.separation_safety_factors().values():
            assert isclose(n0, 20000.0 / ((1 - c) * 10000.0))
        union.set_preload(None)
        assert isclose(union.effective_preload, 25230.0)
        with pytest.raises(ValueError):
            union.set_preload(-1.0)
        with pytest.raises(ValueError):
            union.set_preload(1e9)  # above proof load

    def test_joint_methods_require_plates(self):
        """Member-model accessors raise without plates."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(1000.0, 0.0, 4000.0))
        for attr in ("grip", "bolt_stiffness", "member_stiffness", "joint_constant"):
            with pytest.raises(ValueError):
                getattr(union, attr)
        for method in (union.bolt_tensions, union.separation_safety_factors,
                       union.proof_safety_factors, union.slip_safety_factors):
            with pytest.raises(ValueError):
                method()

    def test_plates_validation(self):
        """Malformed plate stacks raise ValueError."""
        bolt = Bolt("M10", 50.0)
        with pytest.raises(ValueError):
            BoltedUnion(bolt, square_pattern(), plates=[(0.0, "steel"), (20.0, "steel")])
        with pytest.raises(ValueError):
            BoltedUnion(bolt, square_pattern(), plates=[(20.0, "unobtainium"), (20.0, "steel")])
        with pytest.raises(ValueError):
            BoltedUnion(bolt, square_pattern(), plates=[(20.0, "steel")])
        with pytest.raises(ValueError):
            BoltedUnion(bolt, square_pattern(), plates=[(20.0,), (20.0, "steel")])

    def test_safety_factors_unchanged_with_plates(self):
        """The historical safety_factors() ignores the member model."""
        loads = {"forces": (3000.0, 0.0, 5000.0), "moments": (1e5, 0.0, 5e5)}
        bare = BoltedUnion(Bolt("M12", 60.0), square_pattern(), **loads)
        with_plates = BoltedUnion(Bolt("M12", 60.0), square_pattern(),
                                  plates=steel_plates(), **loads)
        assert bare.safety_factors() == with_plates.safety_factors()


class TestMinimumBolt:
    """Test cases for required_stress_area and minimum_bolt sizing."""

    def test_minimum_bolt_pure_shear(self):
        """Slip-critical sizing under pure shear, hand-checked.

        V = 2000 N/bolt, mu = 0.2 -> Fi >= 10000 N ->
        At >= 10000/(0.75*580) = 22.99 mm^2 -> M8 (36.6; M6 = 20.1 fails).
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 0.0))
        assert isclose(union.required_stress_area(mu=0.2), 22.99, rel_tol=1e-3)
        assert union.minimum_bolt(mu=0.2).size == "M8"
        # doubling the design factor doubles the required area -> M10
        assert isclose(union.required_stress_area(mu=0.2, safety_factor=2.0),
                       45.98, rel_tol=1e-3)
        assert union.minimum_bolt(mu=0.2, safety_factor=2.0).size == "M10"
        # the union itself is not modified
        assert union.bolt.size == "M10"

    def test_minimum_bolt_shear_plus_tension(self):
        """Tension adds to the friction demand and the proof check.

        V = 2000, P = 5000 N/bolt: slip needs Fi >= 2000/0.2 + 5000 =
        15000 -> At >= 34.48; proof (C=1): At >= 4*5000/580 = 34.48 -> M8.
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 20000.0))
        assert isclose(union.required_stress_area(mu=0.2), 34.48, rel_tol=1e-3)
        bolt = union.minimum_bolt(mu=0.2)
        assert bolt.size == "M8"
        assert bolt.property_class == "8.8"
        assert bolt.length == 50.0

    def test_minimum_bolt_property_class(self):
        """A stronger property class can shrink the required bolt."""
        union = BoltedUnion(Bolt("M10", 50.0, property_class="4.6"),
                            square_pattern(), forces=(8000.0, 0.0, 0.0))
        weak = union.minimum_bolt(mu=0.2, property_class="4.6")
        strong = union.minimum_bolt(mu=0.2, property_class="12.9")
        assert weak.property_class == "4.6"
        assert strong.property_class == "12.9"
        assert strong.stress_area <= weak.stress_area

    def test_minimum_bolt_with_plates_consistency(self):
        """With plates, the returned bolt passes every joint check >= n."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 20000.0),
                            plates=steel_plates())
        bolt = union.minimum_bolt(mu=0.2)
        union.set_bolt(bolt)
        tol = 1 - 1e-9
        assert all(v >= tol for v in union.slip_safety_factors(mu=0.2).values())
        assert all(v >= tol for v in union.proof_safety_factors().values())
        assert all(v >= tol for v in union.separation_safety_factors().values())

    def test_minimum_bolt_validation(self):
        """Invalid inputs and impossible loads raise ValueError."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            union.minimum_bolt(mu=0.0)
        with pytest.raises(ValueError):
            union.minimum_bolt(mu=0.2, safety_factor=0.0)
        with pytest.raises(ValueError):
            union.minimum_bolt(mu=0.2, property_class="9.9")
        unloaded = BoltedUnion(Bolt("M10", 50.0), square_pattern())
        with pytest.raises(ValueError):
            unloaded.minimum_bolt(mu=0.2)
        huge = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                           forces=(1e8, 0.0, 0.0))
        with pytest.raises(ValueError):
            huge.minimum_bolt(mu=0.2)


class TestBoltLoadCases:
    """CyclicLoadCases mixin on Bolt (Phase 3, D2)."""

    def test_miner_uses_property_class_strength(self):
        """The bolt's S-N curve uses its ISO 898-1 tensile strength."""
        bolt = Bolt(size="M10", length=50.0, property_class="8.8")
        mat = bolt._fatigue_material()
        assert mat.ultimate_strength == pytest.approx(bolt.tensile_strength)
        bolt.add_load_case(200, cycles=1e4).add_load_case(150, 50, cycles=5e4)
        expected = (1e4 / mat.cycles_to_failure(200)
                    + 5e4 / mat.cycles_to_failure(150, 50))
        assert bolt.miner_damage() == pytest.approx(expected)

    def test_load_case_chaining_and_removal(self):
        """add_load_case chains and returns the bolt; remove deletes it."""
        bolt = Bolt(size="M12", length=60.0)
        assert bolt.add_load_case(180, label="a") is bolt
        bolt.remove_load_case("a")
        assert bolt.load_cases == []


class TestBoltMutableInputs:
    """D1: size and property_class are mutable and recompute their lookups."""

    def test_changing_size_recomputes_geometry(self):
        """Reassigning size updates diameter/pitch/stress area from the table."""
        bolt = Bolt(size="M10", length=50.0)
        d10, a10 = bolt.nominal_diameter, bolt.stress_area
        bolt.size = "M16"
        assert bolt.nominal_diameter != pytest.approx(d10)
        assert bolt.nominal_diameter == pytest.approx(Bolt("M16", 50.0).nominal_diameter)
        assert bolt.stress_area != pytest.approx(a10)

    def test_changing_property_class_recomputes_strength(self):
        """Reassigning property_class updates proof/yield/tensile strengths."""
        bolt = Bolt(size="M10", length=50.0, property_class="8.8")
        assert bolt.yield_strength == 640.0
        bolt.property_class = "10.9"
        assert bolt.yield_strength == pytest.approx(
            Bolt("M10", 50.0, property_class="10.9").yield_strength)
        assert bolt.proof_load == pytest.approx(bolt.proof_strength * bolt.stress_area)

    def test_invalid_reassignment_raises(self):
        """Unknown size or property class is rejected on assignment."""
        bolt = Bolt(size="M10", length=50.0)
        with pytest.raises(ValueError):
            bolt.size = "M999"
        with pytest.raises(ValueError):
            bolt.property_class = "42.0"

    def test_custom_bolt_size_is_synthesized(self):
        """A custom bolt reports an M<d>x<p> designation and 'shigley' area."""
        bolt = Bolt(length=50.0, diameter=12.0, pitch=1.25)
        assert bolt.size == "M12x1.25"
        assert bolt.stress_area_method == "shigley"
        with pytest.raises(ValueError):
            bolt.stress_area_method = "table"  # no table size backing it


class TestImperialBolt:
    """Unified inch threads (UNC/UNF) and SAE J429 grades on the same Bolt."""

    def test_unc_geometry_from_table(self):
        """1/2-13 UNC reproduces Shigley Table 8-2 in mm."""
        bolt = Bolt(size="1/2-13", length=50.0, property_class="5")
        assert bolt.thread_series == "UNC"
        assert isclose(bolt.nominal_diameter, 12.7)
        assert isclose(bolt.threads_per_inch, 13.0, rel_tol=1e-3)
        assert isclose(bolt.stress_area, 91.55)  # 0.1419 in^2
        assert isclose(bolt.pitch, 25.4 / 13, rel_tol=1e-4)

    def test_unc_proof_load_matches_shigley(self):
        """1/2-13 grade 5 proof load is Shigley's 12.06 kip = 53.6 kN.

        The end-to-end check on the in->mm conversion chain:
        0.1419 in^2 * 85 ksi = 12.06 kip = 53.65 kN.
        """
        bolt = Bolt(size="1/2-13", length=50.0, property_class="5")
        assert isclose(bolt.proof_load, 53.65e3, rel_tol=2e-3)

    def test_series_suffix_is_optional(self):
        """'1/2-13 UNC' canonicalizes to '1/2-13'; -20 is the UNF thread."""
        assert Bolt(size="1/2-13 UNC", length=50.0).size == "1/2-13"
        assert Bolt(size="1/2-13 unc", length=50.0).stress_area == 91.55
        fine = Bolt(size="1/2-20", length=50.0, property_class="8")
        assert fine.thread_series == "UNF"
        assert fine.stress_area == 103.16  # 0.1599 in^2
        assert fine.nominal_diameter == Bolt(size="1/2-13", length=50.0).nominal_diameter

    def test_sae_grade_is_size_dependent(self):
        """SAE grade 2 derates above 3/4 in: 55 ksi -> 33 ksi proof."""
        small = Bolt(size="3/4-10", length=50.0, property_class="2")
        large = Bolt(size="1-8", length=50.0, property_class="2")
        assert isclose(small.proof_strength, 379.2)  # 55 ksi
        assert isclose(large.proof_strength, 227.5)  # 33 ksi
        # grade 5 switches above 1 in instead
        assert isclose(Bolt(size="1-8", length=50.0, property_class="5").proof_strength,
                       586.1)
        assert isclose(Bolt(size="1-1/8-7", length=50.0, property_class="5").proof_strength,
                       510.2)

    def test_changing_size_recomputes_sae_strength(self):
        """Growing past the grade threshold drops the strength (no caching)."""
        bolt = Bolt(size="3/4-10", length=50.0, property_class="2")
        assert isclose(bolt.proof_strength, 379.2)
        bolt.size = "1-8"
        assert isclose(bolt.proof_strength, 227.5)
        assert bolt.proof_load == pytest.approx(bolt.proof_strength * bolt.stress_area)

    def test_grade_not_defined_for_size(self):
        """A grade that stops at 1 in rejects a larger bolt."""
        bolt = Bolt(size="1-1/4-7", length=60.0, property_class="8.2")
        with pytest.raises(ValueError):
            bolt.proof_strength

    def test_metric_bolts_are_unaffected(self):
        """ISO sizes and classes keep their previous behaviour."""
        bolt = Bolt(size="M10", length=50.0, property_class="8.8")
        assert bolt.thread_series == "ISO metric"
        assert bolt.stress_area == 58.0
        assert bolt.proof_strength == 580.0
        assert Bolt(length=50.0, diameter=10.0, pitch=1.5).thread_series == "custom"

    def test_unknown_designation_and_grade(self):
        """Unknown Unified sizes and SAE grades raise ValueError."""
        with pytest.raises(ValueError):
            Bolt(size="1/2-99", length=50.0)
        with pytest.raises(ValueError):
            Bolt(size="1/2-13", length=50.0, property_class="99")

    def test_minimum_bolt_stays_in_the_same_series(self):
        """Sizing an imperial union returns Unified sizes, not M-sizes.

        V = 2000 N/bolt, mu = 0.2 -> Fi >= 10000 N ->
        At >= 10000/(0.75*586.1) = 22.75 mm^2 -> 5/16-18 (33.81;
        1/4-20 = 20.52 fails).
        """
        union = BoltedUnion(Bolt("1/2-13", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 0.0))
        unc = union.minimum_bolt(mu=0.2, property_class="5")
        assert unc.thread_series == "UNC"
        assert unc.size == "5/16-18"
        fine = BoltedUnion(Bolt("1/2-20", 50.0), square_pattern(),
                           forces=(8000.0, 0.0, 0.0))
        assert fine.minimum_bolt(mu=0.2, property_class="5").thread_series == "UNF"
        # metric unions are unchanged
        metric = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                             forces=(8000.0, 0.0, 0.0))
        assert metric.minimum_bolt(mu=0.2).size == "M8"


class TestPintInputs:
    """Optional pint quantities at the Bolt boundary (plain floats unchanged)."""

    def test_length_accepts_a_quantity(self):
        """length=2 in behaves exactly like length=50.8 mm."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        quantity = Bolt(size="1/2-13", length=2 * ureg.inch)
        plain = Bolt(size="1/2-13", length=50.8)
        assert quantity.length == pytest.approx(plain.length)
        assert quantity.stiffness == pytest.approx(plain.stiffness)

    def test_custom_diameter_and_pitch_accept_quantities(self):
        """An inch diameter/pitch pair lands on the same geometry as mm."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        quantity = Bolt(length=50.0, diameter=0.5 * ureg.inch,
                        pitch=ureg.inch / 13)
        plain = Bolt(length=50.0, diameter=12.7, pitch=25.4 / 13)
        assert quantity.stress_area == pytest.approx(plain.stress_area)

    def test_wrong_dimension_is_rejected(self):
        """A force where a length belongs raises pint's DimensionalityError."""
        pint = pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        with pytest.raises(pint.DimensionalityError):
            Bolt(size="M10", length=5 * ureg.newton)

    def test_plain_floats_still_validate(self):
        """The pint boundary does not weaken the positivity check."""
        with pytest.raises(ValueError):
            Bolt(size="M10", length=-5.0)
        bolt = Bolt(size="M10", length=50.0)
        with pytest.raises(ValueError):
            bolt.length = 0.0


class TestBoltedUnionFatigue:
    """Bolted-joint fatigue (Phase 7.1, Shigley Ch. 8)."""

    def test_fatigue_factor_follows_goodman_preload_line(self):
        """nf matches Eq. 8-38 wired from C, Fi, At, Sut and Se."""
        bolt = Bolt("M10", 50.0, property_class="8.8")
        union = BoltedUnion(bolt, square_pattern(), plates=steel_plates())
        se, p = 140.0, 10000.0
        at = bolt.stress_area
        c = union.joint_constant
        sigma_i = union.effective_preload / at
        sut = bolt.tensile_strength
        sigma_a = c * (p / 2) / at
        sigma_m = sigma_i + c * (p / 2) / at
        expected = se * (sut - sigma_i) / (sut * sigma_a + se * (sigma_m - sigma_i))
        assert union.fatigue_safety_factor(p, endurance_limit=se) == pytest.approx(expected)

    def test_constant_load_gives_infinite_life(self):
        """No alternating component -> infinite fatigue factor."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(), plates=steel_plates())
        assert union.fatigue_safety_factor(5000.0, external_load_min=5000.0) == float("inf")

    def test_requires_plates_and_valid_range(self):
        """No plates, or max < min, is rejected."""
        bolt = Bolt("M10", 50.0)
        with pytest.raises(ValueError):
            BoltedUnion(bolt, square_pattern()).fatigue_safety_factor(1000.0)
        union = BoltedUnion(bolt, square_pattern(), plates=steel_plates())
        with pytest.raises(ValueError):
            union.fatigue_safety_factor(100.0, external_load_min=200.0)
