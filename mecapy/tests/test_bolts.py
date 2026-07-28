"""Tests for bolt module."""

from math import isclose, log, pi, radians, sqrt, tan

import pytest

from mecapy.bolts import (
    Bolt,
    BoltedUnion,
    circular_pattern,
    get_pitch,
    shigley_thread_geometry,
    threaded_length,
)


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

    def test_bending_centroid(self):
        """Pure Mx bending loads bolts proportionally to their y offset."""
        mx = 2e5
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            moments=(mx, 0.0, 0.0), bending_reference="centroid")
        forces = union.bolt_forces()
        # dy = +/-50, sum_dy2 = 4*2500 = 10000 -> axial = +/- mx*50/10000
        expected = mx * 50.0 / 10000.0
        assert isclose(forces[3]["axial"], expected)   # y = 100 (dy = +50), tension
        assert isclose(forces[4]["axial"], expected)
        assert isclose(forces[1]["axial"], -expected)  # y = 0 (dy = -50), compression
        assert isclose(forces[2]["axial"], -expected)

    def test_equilibrium(self):
        """Per-bolt forces re-sum to the applied loads under combined loading.

        Centroid bending only: the extreme-bolt model deliberately leaves
        the compression to plate bearing at the pivot, so its axial loads
        do not balance about the centroid (see
        ``test_bending_extreme_moment_equilibrium``).
        """
        union = BoltedUnion(
            Bolt("M12", 60.0), square_pattern(),
            forces=(3000.0, -2000.0, 5000.0), moments=(1e5, -2e5, 5e5),
            bending_reference="centroid",
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

    def test_bending_extreme(self):
        """Default bending pivots on the farthest bolt: tension only."""
        mx = 2e5
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(), moments=(mx, 0.0, 0.0))
        forces = union.bolt_forces()
        # Pivot is the y = 0 edge; arms are 0 and 100, sum(d^2) = 2*100^2.
        expected = mx * 100.0 / (2 * 100.0 ** 2)
        assert forces[1]["axial"] == 0.0   # y = 0, on the pivot line
        assert forces[2]["axial"] == 0.0
        assert isclose(forces[3]["axial"], expected)  # y = 100, tension
        assert isclose(forces[4]["axial"], expected)
        assert all(e["axial"] >= 0 for e in forces.values())
        assert union.bending_pivots["x"] in (1, 2)
        assert union.bending_pivots["y"] is None

    def test_bending_extreme_negative_moment(self):
        """A negative Mx flips the pivot to the other edge."""
        mx = -2e5
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(), moments=(mx, 0.0, 0.0))
        forces = union.bolt_forces()
        expected = abs(mx) * 100.0 / (2 * 100.0 ** 2)
        assert isclose(forces[1]["axial"], expected)  # y = 0, now in tension
        assert isclose(forces[2]["axial"], expected)
        assert forces[3]["axial"] == 0.0              # y = 100, on the pivot
        assert forces[4]["axial"] == 0.0
        assert union.bending_pivots["x"] in (3, 4)

    def test_bending_extreme_moment_equilibrium(self):
        """Bending tensions balance the applied moment about the pivot line."""
        mx = 5e5
        union = BoltedUnion(
            Bolt("M12", 60.0), circular_pattern(6, 80.0), moments=(mx, 0.0, 0.0)
        )
        forces = union.bolt_forces()
        pivot = union.bending_pivots["x"]
        y_pivot = next(y for number, _x, y in union.positions if number == pivot)
        ys = {number: y for number, _x, y in union.positions}
        assert isclose(forces[pivot]["axial"], 0.0, abs_tol=1e-9)
        assert all(e["axial"] >= 0 for e in forces.values())
        total = sum(
            (ys[number] - y_pivot) * entry["axial"] for number, entry in forces.items()
        )
        assert isclose(total, mx)

    def test_bending_extreme_my_pivot_side(self):
        """Positive My puts the -x bolts in tension, pivoting at +x."""
        my = 2e5
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(), moments=(0.0, my, 0.0))
        forces = union.bolt_forces()
        expected = my * 100.0 / (2 * 100.0 ** 2)
        assert isclose(forces[1]["axial"], expected)  # x = 0, tension
        assert isclose(forces[4]["axial"], expected)
        assert forces[2]["axial"] == 0.0              # x = 100, on the pivot
        assert forces[3]["axial"] == 0.0
        assert union.bending_pivots["y"] in (2, 3)

    def test_axial_breakdown(self):
        """axial_direct + axial_bending_x + axial_bending_y sum to axial."""
        for mode in ("extreme", "centroid"):
            union = BoltedUnion(
                Bolt("M12", 60.0), square_pattern(),
                forces=(3000.0, -2000.0, 5000.0), moments=(1e5, -2e5, 5e5),
                bending_reference=mode,
            )
            for entry in union.bolt_forces().values():
                assert isclose(entry["axial_direct"], 1250.0)
                assert isclose(
                    entry["axial_direct"]
                    + entry["axial_bending_x"]
                    + entry["axial_bending_y"],
                    entry["axial"],
                )

    def test_invalid_bending_reference(self):
        """An unknown bending reference is rejected eagerly."""
        with pytest.raises(ValueError):
            BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                        bending_reference="pivot")
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern())
        with pytest.raises(ValueError):
            union.bending_reference = "extremum"

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
        # bending Mx with all bolts on the x axis, in either mode
        for mode in ("extreme", "centroid"):
            with pytest.raises(ValueError):
                BoltedUnion(
                    bolt, [[1, 0.0, 0.0], [2, 100.0, 0.0]], moments=(1e5, 0.0, 0.0),
                    bending_reference=mode,
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

    def test_plot_distribution_labels_flag(self):
        """labels=False drops the arrow labels but keeps every arrow."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        union = BoltedUnion(
            Bolt("M10", 50.0), square_pattern(),
            forces=(3000.0, 1000.0, 5000.0), moments=(1e5, 0.0, 5e5),
        )

        def counts(**kwargs):
            ax = union.plot_distribution(show=False, **kwargs).axes[0]
            arrows = [t for t in ax.texts
                      if getattr(t, "arrow_patch", None) is not None]
            floating = [t for t in ax.texts
                        if t.get_text().startswith(("Vx", "Vy", "T ="))]
            return len(arrows), len(floating)

        # 4 bolts x (Vx, Vy, T, R) arrows, and 3 labels each when on.
        assert counts() == (16, 12)
        assert counts(labels=False) == (16, 0)

    def test_plot_distribution_arrows_stay_inside_the_view(self):
        """Arrow tips are inside the axes: they are clipped, unlike text."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        # Bolts at the corners with shear pushing outward: sizing the view
        # from the bolt positions alone used to clip whole arrows away.
        union = BoltedUnion(
            Bolt("M12", 60.0), square_pattern(),
            forces=(3000.0, 2000.0, 12000.0), moments=(4e5, -2e5, 8e5),
        )
        ax = union.plot_distribution(show=False).axes[0]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        tips = [t.xy for t in ax.texts
                if getattr(t, "arrow_patch", None) is not None]
        assert tips
        assert all(x0 <= tx <= x1 and y0 <= ty <= y1 for tx, ty in tips)


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


class TestBoltSegmentedStiffness:
    """Test cases for Shigley Eq. 8-17 bolt stiffness in a grip."""

    def test_segmented_stiffness_m10(self):
        """M10x50 in a 40 mm grip: kb = Ad*At*E/(Ad*lt + At*ld) longhand.

        LT = 2*10 + 6 = 26, so ld = 50 - 26 = 24 and lt = 40 - 24 = 16.
        Ad = pi*10^2/4 = 78.5398, At = 58, E = 210000 MPa
        -> kb = 78.5398*58*210000/(78.5398*16 + 58*24) = 361172.5 N/mm.
        """
        bolt = Bolt("M10", 50.0)
        assert bolt.threaded_length == 26.0
        assert bolt.shank_length == 24.0
        ad = pi * 10.0 ** 2 / 4
        at, e = 58.0, 210000.0
        expected = ad * at * e / (ad * 16.0 + at * 24.0)
        assert isclose(bolt.segmented_stiffness(40.0), expected, rel_tol=1e-9)
        assert isclose(bolt.segmented_stiffness(40.0), 3.61172e5, rel_tol=1e-4)

    def test_segmented_is_stiffer_than_free_length(self):
        """The segmented model beats the single-area model over the grip."""
        bolt = Bolt("M10", 50.0)
        single_area = bolt.stress_area * bolt.elastic_modulus / 40.0
        assert bolt.segmented_stiffness(40.0) > single_area

    def test_fully_unthreaded_grip(self):
        """A grip inside the shank reduces to kb = Ad*E/grip."""
        bolt = Bolt("M10", 50.0)  # shank 24 mm
        expected = bolt.nominal_area * bolt.elastic_modulus / 20.0
        assert isclose(bolt.segmented_stiffness(20.0), expected, rel_tol=1e-9)

    def test_fully_threaded_bolt(self):
        """A bolt shorter than its LT has no shank: kb = At*E/grip."""
        bolt = Bolt("M20", 40.0)  # LT = 46 > 40
        assert bolt.shank_length == 0.0
        expected = bolt.stress_area * bolt.elastic_modulus / 30.0
        assert isclose(bolt.segmented_stiffness(30.0), expected, rel_tol=1e-9)

    def test_segmented_stiffness_validation(self):
        """Non-physical grips raise ValueError."""
        bolt = Bolt("M10", 50.0)
        with pytest.raises(ValueError):
            bolt.segmented_stiffness(0.0)
        with pytest.raises(ValueError):
            bolt.segmented_stiffness(-1.0)
        with pytest.raises(ValueError):
            bolt.segmented_stiffness(60.0)  # longer than the bolt


class TestThreadedLength:
    """Test cases for the Shigley Table 8-7 threaded-length rule."""

    def test_metric_length_bands(self):
        """LT = 2d + 6 / 12 / 25 mm across the three length bands."""
        assert threaded_length(10.0, 50.0) == 2 * 10.0 + 6.0
        assert threaded_length(10.0, 125.0) == 2 * 10.0 + 6.0  # band edge
        assert threaded_length(10.0, 150.0) == 2 * 10.0 + 12.0
        assert threaded_length(10.0, 200.0) == 2 * 10.0 + 12.0  # band edge
        assert threaded_length(10.0, 250.0) == 2 * 10.0 + 25.0

    def test_unified_length_bands(self):
        """Unified: LT = 2d + 1/4 in up to 6 in, then 2d + 1/2 in."""
        assert threaded_length(12.7, 50.0, series="UNC") == 2 * 12.7 + 6.35
        assert threaded_length(12.7, 200.0, series="UNF") == 2 * 12.7 + 12.7

    def test_short_bolt_is_fully_threaded(self):
        """A bolt shorter than its own LT gets LT >= length, uncapped."""
        assert threaded_length(20.0, 40.0) == 46.0

    def test_threaded_length_validation(self):
        """Non-physical inputs and out-of-table diameters raise ValueError."""
        with pytest.raises(ValueError):
            threaded_length(0.0, 50.0)
        with pytest.raises(ValueError):
            threaded_length(10.0, 0.0)
        with pytest.raises(ValueError):
            threaded_length(60.0, 100.0)  # beyond the 48 mm metric table


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

        The bolt is the segmented Eq. 8-17 spring: LT = 2*10 + 6 = 26,
        ld = 50 - 26 = 24, lt = 40 - 24 = 16, Ad = 78.5398, At = 58
        -> kb = 361172.5 N/mm and C = kb/(kb + km) ~= 0.1689.
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            plates=steel_plates())
        assert union.grip == 40.0
        tan30 = tan(radians(30))
        ln_arg = (2 * tan30 * 20.0 + 5.0) * 25.0 / ((2 * tan30 * 20.0 + 25.0) * 5.0)
        expected_k1 = tan30 * pi * 210000.0 * 10.0 / log(ln_arg)
        assert isclose(union.member_stiffness, expected_k1 / 2, rel_tol=1e-9)
        assert isclose(union.member_stiffness, 1.7769e6, rel_tol=1e-3)
        # kb = Ad*At*E / (Ad*lt + At*ld), Shigley eq. 8-17
        ad = pi * 10.0 ** 2 / 4
        expected_kb = ad * 58.0 * 210000.0 / (ad * 16.0 + 58.0 * 24.0)
        assert isclose(union.bolt_stiffness, expected_kb, rel_tol=1e-9)
        assert isclose(union.bolt_stiffness, 361172.5, rel_tol=1e-6)
        expected_c = expected_kb / (expected_kb + expected_k1 / 2)
        assert isclose(union.joint_constant, expected_c, rel_tol=1e-9)
        assert isclose(union.joint_constant, 0.1689, rel_tol=1e-3)

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

    def test_bolt_stiffness_delegates_to_segmented_model(self):
        """The union's kb is the bolt's Eq. 8-17 spring over the grip."""
        bolt = Bolt("M10", 50.0)
        union = BoltedUnion(bolt, square_pattern(), plates=steel_plates())
        assert union.bolt_stiffness == bolt.segmented_stiffness(union.grip)
        # and the sizing path must agree, or minimum_bolt desyncs from C
        assert isclose(union._joint_constant_for(bolt), union.joint_constant,
                       rel_tol=1e-12)

    def test_bolt_tensions_separation_proof(self):
        """Preload-aware distribution and factors, hand-checked chain.

        M10 8.8: Fi = 0.75*580*58 = 25230 N; Fz = 40000 -> P = 10000 N.
        With the segmented C = 0.1689: Fb = 26919 N, Fm = -16919 N
        (still clamped), n0 = 3.036 and nL = 4.978.
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
            assert isclose(n0, 3.036, rel_tol=1e-3)
        for nl in union.proof_safety_factors().values():
            assert isclose(nl, (580.0 * 58.0 - fi) / (c * 10000.0))
            assert isclose(nl, 4.978, rel_tol=1e-3)

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
        for attr in ("grip", "effective_grip", "bolt_stiffness",
                     "member_stiffness", "joint_constant"):
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


class TestTappedJoint:
    """Test cases for blind/tapped-hole joints (Shigley Sec. 8-5)."""

    def test_effective_grip_thin_tapped_member(self):
        """t2 < d: l' = h + t2/2, while grip stays the physical stack."""
        union = BoltedUnion(Bolt("M10", 24.0), square_pattern(),
                            plates=[(20.0, "steel"), (6.0, "steel")], tapped=True)
        assert union.grip == 26.0
        assert union.effective_grip == 20.0 + 6.0 / 2  # t2 = 6 < d = 10

    def test_effective_grip_thick_tapped_member(self):
        """t2 >= d: the effective addition saturates at d/2."""
        union = BoltedUnion(Bolt("M10", 30.0), square_pattern(),
                            plates=[(20.0, "steel"), (20.0, "steel")], tapped=True)
        assert union.grip == 40.0
        assert union.effective_grip == 20.0 + 10.0 / 2  # d/2

    def test_through_joint_effective_grip_is_the_grip(self):
        """Nothing changes for a nutted joint -- the regression guard."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            plates=steel_plates())
        assert union.effective_grip == union.grip
        assert union.tapped is False

    def test_tapped_members_are_stiffer(self):
        """The shorter effective grip stiffens the members.

        Same physical stack, but the cones span l' = 25 instead of
        40 mm. Only km is compared: a tapped joint needs a shorter bolt
        (it must not bottom out), so kb and C would be comparing two
        different bolts.
        """
        plates = [(20.0, "steel"), (20.0, "steel")]
        through = BoltedUnion(Bolt("M10", 40.0), square_pattern(), plates=plates)
        tapped = BoltedUnion(Bolt("M10", 30.0), square_pattern(),
                             plates=plates, tapped=True)
        assert tapped.effective_grip < through.effective_grip
        assert tapped.member_stiffness > through.member_stiffness

    def test_tapped_bolt_stiffness_uses_effective_grip(self):
        """kb is the segmented spring over l', not over the full stack."""
        bolt = Bolt("M10", 30.0)
        union = BoltedUnion(bolt, square_pattern(),
                            plates=[(20.0, "steel"), (20.0, "steel")], tapped=True)
        assert union.effective_grip == 25.0 != union.grip
        assert union.bolt_stiffness == bolt.segmented_stiffness(25.0)

    def test_tapped_member_stiffness_longhand(self):
        """km over the effective stack, recomputed frustum by frustum.

        l' = 20 + 5 = 25 mm, midplane at 12.5. Each half is one steel
        frustum with t = 12.5 and D = dw = 15 (the mirrored stack is all
        steel), so km = k1/2 with k1 the standard frustum formula.
        """
        union = BoltedUnion(Bolt("M10", 30.0), square_pattern(),
                            plates=[(20.0, "steel"), (20.0, "steel")], tapped=True)
        tan30 = tan(radians(30))
        t, small_d, d = 12.5, 15.0, 10.0
        ln_arg = ((2 * tan30 * t + small_d - d) * (small_d + d)
                  / ((2 * tan30 * t + small_d + d) * (small_d - d)))
        k1 = tan30 * pi * 210000.0 * d / log(ln_arg)
        assert isclose(union.member_stiffness, k1 / 2, rel_tol=1e-9)

    def test_tapped_bolt_length_validation(self):
        """The bolt must reach into the tapped member without bottoming."""
        plates = [(20.0, "steel"), (10.0, "steel")]
        with pytest.raises(ValueError):  # does not reach in (L <= 20)
            BoltedUnion(Bolt("M10", 20.0), square_pattern(),
                        plates=plates, tapped=True)
        with pytest.raises(ValueError):  # bottoms out (L > 30)
            BoltedUnion(Bolt("M10", 35.0), square_pattern(),
                        plates=plates, tapped=True)
        union = BoltedUnion(Bolt("M10", 26.0), square_pattern(),
                            plates=plates, tapped=True)
        assert union.tapped is True

    def test_single_member_still_rejected(self):
        """Tapped or not, a joint needs at least two members."""
        with pytest.raises(ValueError):
            BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                        plates=[(20.0, "steel")], tapped=True)

    def test_tapped_is_settable(self):
        """Toggling tapped re-validates the bolt and moves l'."""
        union = BoltedUnion(Bolt("M10", 30.0), square_pattern(),
                            plates=[(20.0, "steel"), (20.0, "steel")], tapped=True)
        assert union.effective_grip == 25.0
        with pytest.raises(ValueError):
            union.tapped = False  # 30 mm bolt cannot span the 40 mm grip
        assert union.tapped is True

    def test_clearing_plates_clears_tapped(self):
        """Removing the member model resets the tapped flag."""
        union = BoltedUnion(Bolt("M10", 30.0), square_pattern(),
                            plates=[(20.0, "steel"), (20.0, "steel")], tapped=True)
        union.set_plates(None)
        assert union.tapped is False


class TestMemberChecks:
    """Test cases for bearing, clamp state and tear-out on the members."""

    def test_bearing_stress_per_member(self):
        """sigma = V/(d*t) against the plate yield, hand-checked.

        Fx = 8000 N over 4 bolts -> V = 2000 N; d = 10, t = 20
        -> sigma = 10 MPa and n = 250/10 = 25 on both steel plates.
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 0.0), plates=steel_plates())
        entry = union.bearing_stresses()[1]
        assert isclose(entry["shear"], 2000.0)
        assert len(entry["members"]) == 2
        for member in entry["members"]:
            assert isclose(member["stress"], 10.0)
            assert isclose(member["safety_factor"], 25.0)
        assert isclose(entry["safety_factor"], 25.0)
        assert isclose(union.bearing_safety_factors()[1], 25.0)

    def test_bearing_without_shear_is_infinite(self):
        """A bolt carrying no shear cannot crush its hole."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(0.0, 0.0, 8000.0), plates=steel_plates())
        assert union.bearing_safety_factors()[1] == float("inf")

    def test_clamp_state_while_clamped(self):
        """Clamp is -Fm, and the head bears with the full bolt tension.

        Fz = 40000 -> P = 10000 N, C = 0.1689, Fi = 25230:
        Fb = 26919 N, Fm = -16919 N so clamp = 16919 N.
        Washer annulus = pi/4*(15^2 - 10^2) = 98.17 mm^2
        -> p = 26919/98.17 = 274.2 MPa, n = 250/274.2 = 0.912. That is
        below 1: bearing a full-preload M10 8.8 straight onto mild steel
        really does yield the plate, which is why washers exist.
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(0.0, 0.0, 40000.0), plates=steel_plates())
        entry = union.clamp_states()[1]
        assert entry["separated"] is False
        assert isclose(entry["clamp"], -entry["member_force"])
        assert isclose(entry["clamp"], 16919.0, rel_tol=1e-3)
        area = pi / 4 * (15.0 ** 2 - 10.0 ** 2)
        assert isclose(entry["washer_pressure"], entry["bolt_tension"] / area)
        assert isclose(entry["washer_pressure"], 274.2, rel_tol=1e-3)
        assert isclose(entry["washer_safety_factor"], 0.912, rel_tol=1e-3)

    def test_clamp_state_detects_separation(self):
        """Once Fm >= 0 the members have decompressed.

        Separation agrees with the existing n0 < 1 criterion.
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(0.0, 0.0, 200000.0), plates=steel_plates())
        entry = union.clamp_states()[1]
        assert entry["separated"] is True
        assert entry["clamp"] == 0.0
        assert union.separation_safety_factors()[1] < 1.0

    def test_minimum_edge_distance(self):
        """e = d/2 + n*V/(2*t*0.577*Sy), hand-checked.

        V = 2000, t = 20, Sy = 250, n = 2
        -> e = 5 + 2*2000/(2*20*0.577*250) = 5 + 0.693 = 5.693 mm.
        """
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 0.0), plates=steel_plates())
        entry = union.minimum_edge_distances(safety_factor=2.0)[1]
        expected = 5.0 + 2.0 * 2000.0 / (2 * 20.0 * 0.577 * 250.0)
        assert isclose(entry["edge_distance"], expected, rel_tol=1e-9)
        assert isclose(entry["edge_distance"], 5.693, rel_tol=1e-3)

    def test_edge_distance_scales_with_safety_factor(self):
        """The tear-out margin above d/2 is linear in the design factor."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 0.0), plates=steel_plates())
        one = union.minimum_edge_distances(1.0)[1]["edge_distance"]
        three = union.minimum_edge_distances(3.0)[1]["edge_distance"]
        assert isclose(three - 5.0, 3 * (one - 5.0), rel_tol=1e-9)

    def test_edge_distance_without_shear_is_the_hole(self):
        """No shear means nothing to tear out: e = d/2."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(0.0, 0.0, 8000.0), plates=steel_plates())
        assert union.minimum_edge_distances()[1]["edge_distance"] == 5.0

    def test_hole_diameter_default_and_setter(self):
        """The hole defaults to the bolt diameter and shrinks the annulus."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(0.0, 0.0, 40000.0), plates=steel_plates())
        assert union.hole_diameter == 10.0
        tight = union.clamp_states()[1]["washer_pressure"]
        union.hole_diameter = 11.0
        assert union.clamp_states()[1]["washer_pressure"] > tight
        union.hole_diameter = None
        assert union.hole_diameter == 10.0

    def test_member_checks_validation(self):
        """Bad inputs and a missing member model raise ValueError."""
        bare = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                           forces=(8000.0, 0.0, 4000.0))
        for method in (bare.bearing_stresses, bare.bearing_safety_factors,
                       bare.clamp_states, bare.minimum_edge_distances):
            with pytest.raises(ValueError):
                method()
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 0.0), plates=steel_plates())
        with pytest.raises(ValueError):
            union.minimum_edge_distances(safety_factor=0.0)
        with pytest.raises(ValueError):
            union.hole_diameter = 9.0   # smaller than the bolt
        with pytest.raises(ValueError):
            union.hole_diameter = 15.0  # no washer face left


class TestJointReport:
    """Test cases for joint_report, describe and plot_tension."""

    def loaded_union(self):
        """A 4-bolt joint with plates and a full six-component load."""
        return BoltedUnion(
            Bolt("M10", 50.0), square_pattern(),
            forces=(8000.0, 0.0, 20000.0), moments=(2e5, 0.0, 4e5),
            plates=steel_plates(),
        )

    def test_report_matches_the_accessors(self):
        """joint_report re-exports the same numbers, not new ones."""
        union = self.loaded_union()
        report = union.joint_report()
        assert report["n_bolts"] == 4
        assert report["joint_constant"] == union.joint_constant
        assert report["bolt_stiffness"] == union.bolt_stiffness
        assert report["member_stiffness"] == union.member_stiffness
        assert report["effective_grip"] == union.grip
        assert report["preload"] == union.effective_preload
        assert report["tapped"] is False
        tensions = union.bolt_tensions()
        edges = union.minimum_edge_distances()
        for number, row in report["bolts"].items():
            assert row["bolt_tension"] == tensions[number]["bolt_tension"]
            assert row["member_force"] == tensions[number]["member_force"]
            assert row["separation"] == union.separation_safety_factors()[number]
            assert row["bearing"] == union.bearing_safety_factors()[number]
            assert row["edge_distance"] == edges[number]["edge_distance"]

    def test_report_without_plates_degrades(self):
        """No member model: stiffness entries are None, forces remain."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 4000.0))
        report = union.joint_report()
        for key in ("grip", "effective_grip", "bolt_stiffness",
                    "member_stiffness", "joint_constant", "preload"):
            assert report[key] is None
        assert len(report["bolts"]) == 4
        for row in report["bolts"].values():
            assert "bolt_tension" not in row
            assert isclose(row["shear"], 2000.0)

    def test_describe_is_a_returned_string(self):
        """describe returns the report; it does not print it."""
        union = self.loaded_union()
        text = union.describe()
        assert isinstance(text, str)
        assert "joint constant (C)" in text
        assert "bolt stiffness (kb)" in text
        for number in (1, 2, 3, 4):
            assert f"{number:>3}" in text

    def test_describe_without_plates_does_not_raise(self):
        """A bare bolt group still describes its force distribution."""
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(8000.0, 0.0, 4000.0))
        text = union.describe()
        assert "no plates defined" in text
        assert "joint constant" not in text

    def test_describe_flags_a_tapped_joint(self):
        """The effective grip line is tagged when the joint is tapped."""
        union = BoltedUnion(Bolt("M10", 30.0), square_pattern(),
                            plates=[(20.0, "steel"), (20.0, "steel")], tapped=True)
        assert "(tapped)" in union.describe()

    def test_plot_tension_smoke(self):
        """plot_tension returns a Figure without showing it."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        fig = self.loaded_union().plot_tension(show=False)
        assert isinstance(fig, Figure)

    def test_plot_tension_labels_flag(self):
        """labels=False drops the per-bolt annotations."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        union = self.loaded_union()
        with_labels = union.plot_tension(show=False, labels=True).axes[0]
        without = union.plot_tension(show=False, labels=False).axes[0]
        assert len(with_labels.texts) > len(without.texts)

    def test_plot_tension_requires_plates(self):
        """The tension plot needs a member model."""
        pytest.importorskip("matplotlib")
        union = BoltedUnion(Bolt("M10", 50.0), square_pattern(),
                            forces=(0.0, 0.0, 4000.0))
        with pytest.raises(ValueError):
            union.plot_tension(show=False)


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


class TestBoltedUnionPintInputs:
    """Optional pint quantities at the BoltedUnion boundary."""

    def test_positions_forces_and_moments_accept_quantities(self):
        """An imperial load case lands on the same state as the mm/N one."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        quantity = BoltedUnion(
            Bolt("M12", 60.0),
            [[1, 0.0, 0.0], [2, 4 * ureg.inch, 0.0]],
            forces=(2.5 * ureg.kN, 0, 30 * ureg.kN),
            moments=(0, 0, 3.2 * ureg.kN * ureg.m),
        )
        plain = BoltedUnion(
            Bolt("M12", 60.0),
            [[1, 0.0, 0.0], [2, 101.6, 0.0]],
            forces=(2500.0, 0.0, 30000.0),
            moments=(0.0, 0.0, 3.2e6),
        )
        assert quantity.positions == plain.positions
        assert quantity.forces == pytest.approx(plain.forces)
        assert quantity.moments == pytest.approx(plain.moments)
        assert quantity.centroid == pytest.approx(plain.centroid)
        for number, entry in quantity.bolt_forces().items():
            assert entry["shear_magnitude"] == pytest.approx(
                plain.bolt_forces()[number]["shear_magnitude"]
            )

    def test_plate_thickness_and_preload_accept_quantities(self):
        """Grip and preload convert from inches and kip."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        union = BoltedUnion(
            Bolt("M12", 60.0),
            square_pattern(),
            plates=[(0.75 * ureg.inch, "steel"), (20.0, "steel")],
            preload=8 * ureg.kip,
        )
        assert union.grip == pytest.approx(0.75 * 25.4 + 20.0)
        assert union.preload == pytest.approx(8 * 4448.2216152605)

    def test_circular_pattern_radius_accepts_a_quantity(self):
        """circular_pattern(radius=4 in) equals radius=101.6 mm."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        quantity = circular_pattern(6, 4 * ureg.inch)
        plain = circular_pattern(6, 101.6)
        assert [row[0] for row in quantity] == [row[0] for row in plain]
        for got, want in zip(quantity, plain):
            assert got[1:] == pytest.approx(want[1:])

    def test_cone_angle_and_centroid_accept_quantities(self):
        """The angle converts to degrees, the centroid override to mm."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        union = BoltedUnion(Bolt("M12", 60.0), square_pattern())
        union.cone_angle_deg = 0.5 * ureg.radian
        assert union.cone_angle_deg == pytest.approx(28.6478897565)
        union.centroid = (1 * ureg.cm, 0)
        assert union.centroid == pytest.approx((10.0, 0.0))

    def test_fatigue_loads_accept_quantities(self):
        """Fatigue load bounds in kN match the same loads in N."""
        pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        union = BoltedUnion(
            Bolt("M12", 60.0), square_pattern(),
            plates=[(20.0, "steel"), (20.0, "steel")],
        )
        assert union.fatigue_safety_factor(
            2 * ureg.kN, 0.5 * ureg.kN
        ) == pytest.approx(union.fatigue_safety_factor(2000.0, 500.0))

    def test_wrong_dimension_is_rejected(self):
        """A force where a length belongs raises pint's DimensionalityError."""
        pint = pytest.importorskip("pint")
        from mecapy.utils.units import ureg

        with pytest.raises(pint.DimensionalityError):
            BoltedUnion(Bolt("M12", 60.0), [[1, 0.0, 0.0], [2, 5 * ureg.newton, 0.0]])

    def test_plain_floats_still_validate(self):
        """The pint boundary does not weaken the existing checks."""
        bolt = Bolt("M12", 60.0)
        with pytest.raises(ValueError):
            BoltedUnion(bolt, square_pattern(),
                        plates=[(-1.0, "steel"), (20.0, "steel")])
        with pytest.raises(ValueError):
            BoltedUnion(bolt, square_pattern(), preload=-100.0)


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
