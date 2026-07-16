"""Tests for the weld module (Weld, geometry and WeldedUnion)."""

import math
from math import isclose, pi, sqrt

import pytest
from mecapy.welds import (
    Weld,
    WeldLine,
    WeldCircle,
    WeldedUnion,
    get_electrode,
)


class TestWeldGeometry:
    """Weld path geometry (WeldLine, WeldCircle)."""

    def test_line_length_and_centroid(self):
        """A line's length is the distance and its centroid the midpoint."""
        line = WeldLine((0, 0), (0, 100))
        assert isclose(line.length, 100.0)
        assert line.centroid == (0.0, 50.0)

    def test_line_unit_inertia(self):
        """Vertical line length d: Ix_own = d^3/12 (Shigley T9-2)."""
        line = WeldLine((0, 0), (0, 100))
        ix, iy = line.unit_inertia()
        assert isclose(ix, 100 ** 3 / 12)
        assert isclose(iy, 0.0, abs_tol=1e-9)

    def test_circle_unit_inertia(self):
        """Circle radius r: Iu = pi*r^3, Ju = 2*pi*r^3 (Shigley T9-1)."""
        circle = WeldCircle((0, 0), 25)
        ix, iy = circle.unit_inertia()
        assert isclose(ix, pi * 25 ** 3)
        assert isclose(iy, pi * 25 ** 3)
        assert isclose(circle.length, 2 * pi * 25)

    def test_geometry_validation(self):
        """Degenerate geometry raises."""
        with pytest.raises(ValueError):
            WeldLine((0, 0), (0, 0))
        with pytest.raises(ValueError):
            WeldCircle((0, 0), 0)


class TestWeld:
    """Single-weld geometry, throat and electrode strength."""

    def test_throat_fillet_and_butt(self):
        """Fillet throat = 0.707*size; butt throat = size."""
        fillet = Weld(WeldLine((0, 0), (0, 100)), size=6.0)
        assert isclose(fillet.throat, 0.707 * 6.0)
        assert isclose(fillet.throat_area, 0.707 * 6.0 * 100.0)
        butt = Weld(WeldLine((0, 0), (0, 100)), size=6.0, weld_type="butt")
        assert isclose(butt.throat, 6.0)

    def test_electrode_allowable(self):
        """E70xx: tensile 482 MPa, allowable = 0.30*482 (Shigley T9-4)."""
        weld = Weld(WeldLine((0, 0), (0, 100)), size=6.0, electrode="E70")
        assert isclose(weld.electrode_tensile, 482.0)
        assert isclose(weld.allowable_stress, 0.30 * 482.0)

    def test_unit_inertia_about_parallel_axis(self):
        """Parallel-axis shift adds L*(offset)^2."""
        weld = Weld(WeldLine((0, 0), (0, 100)), size=6.0)
        # About its own centroid (0, 50): pure own inertia.
        ix, _ = weld.unit_inertia_about(0, 50)
        assert isclose(ix, 100 ** 3 / 12)
        # Shifted 50 mm in y: add L*50^2.
        ix2, _ = weld.unit_inertia_about(0, 0)
        assert isclose(ix2, 100 ** 3 / 12 + 100 * 50 ** 2)

    def test_weld_validation(self):
        """Bad path, weld type, size or electrode raise."""
        with pytest.raises(ValueError):
            Weld("not a path", size=6.0)
        with pytest.raises(ValueError):
            Weld(WeldLine((0, 0), (0, 100)), size=6.0, weld_type="spot")
        with pytest.raises(ValueError):
            Weld(WeldLine((0, 0), (0, 100)), size=-1.0)
        with pytest.raises(ValueError):
            Weld(WeldLine((0, 0), (0, 100)), size=6.0, electrode="E999")
        # No size set -> asking for the throat raises.
        with pytest.raises(ValueError):
            Weld(WeldLine((0, 0), (0, 100))).throat

    def test_electrode_lookup_normalization(self):
        """'E70', 'e70' and 'E70xx' resolve to the same entry."""
        assert get_electrode("E70") is get_electrode("E70xx")
        assert get_electrode("e70")["tensile_strength"] == 482.0
        with pytest.raises(ValueError):
            get_electrode("E55")


def two_vertical_lines(size=6.0):
    """Two parallel vertical welds 50 mm apart (helper, not a fixture)."""
    return [
        Weld(WeldLine((0, 0), (0, 100)), size=size),
        Weld(WeldLine((50, 0), (50, 100)), size=size),
    ]


class TestWeldedUnion:
    """Weld-group section properties, stress and safety."""

    def test_centroid_length_weighted(self):
        """Two equal vertical welds -> centroid midway between them."""
        u = WeldedUnion(two_vertical_lines())
        assert u.centroid == pytest.approx((25.0, 50.0))
        assert isclose(u.total_length, 200.0)

    def test_unit_moments(self):
        """Iux/Iuy/Ju via parallel axis for the two-line group."""
        u = WeldedUnion(two_vertical_lines())
        iux, iuy = u.unit_second_moment
        # Each line: Iux_own = 100^3/12; both lie on centroidal x, so no
        # shift in x. Iux = 2 * 100^3/12.
        assert isclose(iux, 2 * 100 ** 3 / 12)
        # Iuy: each line horizontal offset 25 from centroid -> 100*25^2.
        assert isclose(iuy, 2 * 100 * 25 ** 2)
        assert isclose(u.unit_polar_moment, iux + iuy)

    def test_single_line_direct_shear(self):
        """Direct shear: tau = (Fy/L)/throat, sigma_eq = sqrt(3)*tau."""
        weld = Weld(WeldLine((0, 0), (0, 100)), size=6.0)
        u = WeldedUnion([weld], forces=(0, -6000, 0))
        throat = 0.707 * 6.0
        tau = 6000 / 100 / throat
        _, _, sigma_eq = u.max_stress()
        assert isclose(sigma_eq, sqrt(3) * tau, rel_tol=1e-9)

    def test_torsion_peaks_at_farthest_point(self):
        """Under Mz the most stressed point is farthest from the centroid."""
        u = WeldedUnion(two_vertical_lines(), moments=(0, 0, 500000))
        _, (px, py), sigma = u.max_stress()
        cx, cy = u.centroid
        # Farthest sample point is at a corner (max distance from centroid).
        r = math.hypot(px - cx, py - cy)
        assert sigma > 0
        assert r == pytest.approx(math.hypot(25, 50), abs=1.0)

    def test_bulk_size_change_rescales_stress(self):
        """Doubling the size doubles the throat and halves every stress."""
        u = WeldedUnion(two_vertical_lines(size=6.0), forces=(0, -8000, 0))
        _, _, s1 = u.max_stress()
        u.size = 12.0
        assert u.size == 12.0
        _, _, s2 = u.max_stress()
        assert isclose(s2, s1 / 2, rel_tol=1e-9)

    def test_safety_factor_vs_allowable(self):
        """SF = 0.30*FEXX / peak stress."""
        weld = Weld(WeldLine((0, 0), (0, 100)), size=6.0, electrode="E70")
        u = WeldedUnion([weld], forces=(0, -6000, 0))
        _, _, sigma_eq = u.max_stress()
        sf = u.safety_factors()[0]
        assert isclose(sf, (0.30 * 482.0) / sigma_eq, rel_tol=1e-9)

    def test_circle_group(self):
        """A single circular weld: Ju = 2*pi*r^3, torsion gives shear."""
        u = WeldedUnion(
            [Weld(WeldCircle((0, 0), 25), size=6.0)],
            moments=(0, 0, 300000),
        )
        assert isclose(u.unit_polar_moment, 2 * pi * 25 ** 3)
        _, _, sigma = u.max_stress()
        assert sigma > 0

    def test_union_validation(self):
        """Empty list, non-Weld items or bad load vectors raise."""
        with pytest.raises(ValueError):
            WeldedUnion([])
        with pytest.raises(ValueError):
            WeldedUnion(["nope"])
        with pytest.raises(ValueError):
            WeldedUnion(two_vertical_lines(), forces=(1, 2))
        with pytest.raises(ValueError):
            WeldedUnion(two_vertical_lines(), moments=(1, 2))

    def test_mixed_size_property_raises(self):
        """The 'size' getter needs a single common size."""
        welds = [
            Weld(WeldLine((0, 0), (0, 100)), size=6.0),
            Weld(WeldLine((50, 0), (50, 100)), size=8.0),
        ]
        u = WeldedUnion(welds)
        with pytest.raises(ValueError):
            _ = u.size
        # Setting size makes them uniform again.
        u.size = 10.0
        assert u.size == 10.0


class TestWeldPlot:
    """Smoke test for the matplotlib plot."""

    def test_plot_returns_figure(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        u = WeldedUnion(two_vertical_lines(), forces=(0, -8000, 0),
                        moments=(0, 0, 400000))
        fig = u.plot_distribution(show=False)
        assert isinstance(fig, Figure)
