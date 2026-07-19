"""Tests for shaft module."""

import math

import pytest

from mecapy import MechaElement
from mecapy.shafts import Shaft, SteppedShaft
from mecapy.shafts.kt_data import get_kt_groove, get_kt_shoulder_fillet


class TestShaft:
    """Test cases for Shaft class."""

    def test_shaft_creation(self):
        """Test creating a shaft object."""
        shaft = Shaft(diameter=25.0, length=500.0, material="steel")
        assert shaft.diameter == 25.0
        assert shaft.length == 500.0
        assert shaft.material == "steel"

    def test_shaft_is_mecha_element(self):
        """Shaft inherits from MechaElement."""
        shaft = Shaft(diameter=25.0, length=500.0)
        assert isinstance(shaft, MechaElement)

    def test_polar_moment(self):
        """Polar moment of area is pi * d^4 / 32."""
        shaft = Shaft(diameter=20.0, length=500.0)
        assert shaft.polar_moment == pytest.approx(math.pi * 20.0 ** 4 / 32)

    def test_torsional_stress(self):
        """Torsional shear stress is T*r/J."""
        shaft = Shaft(diameter=20.0, length=500.0)
        torque = 1e5  # N*mm
        expected = torque * 10.0 / (math.pi * 20.0 ** 4 / 32)
        assert shaft.torsional_stress(torque) == pytest.approx(expected)

    def test_shaft_repr(self):
        """Test shaft string representation."""
        shaft = Shaft(diameter=30.0, length=1000.0, material="steel")
        assert "Shaft" in repr(shaft)
        assert "30.0mm" in repr(shaft)


class TestKtData:
    """Test cases for the Kt approximation helpers."""

    def test_shoulder_kt_at_least_one(self):
        """Shoulder fillet Kt is never below 1."""
        assert get_kt_shoulder_fillet(30, 20, 3) >= 1.0

    def test_shoulder_kt_decreases_with_larger_radius(self):
        """A larger fillet radius gives a smaller Kt (smoother transition)."""
        kt_small_r = get_kt_shoulder_fillet(30, 20, 1)
        kt_large_r = get_kt_shoulder_fillet(30, 20, 5)
        assert kt_large_r < kt_small_r

    def test_groove_kt_exceeds_shoulder_kt_for_same_geometry(self):
        """A groove concentrates stress more than an equivalent shoulder fillet."""
        assert get_kt_groove(30, 20, 2) > get_kt_shoulder_fillet(30, 20, 2)

    def test_requires_D_greater_than_d(self):
        """D <= d is rejected."""
        with pytest.raises(ValueError):
            get_kt_shoulder_fillet(20, 20, 2)

    def test_requires_positive_radius(self):
        """A non-positive radius is rejected."""
        with pytest.raises(ValueError):
            get_kt_shoulder_fillet(30, 20, 0)

    def test_unknown_loading_rejected(self):
        """An unrecognized loading string is rejected."""
        with pytest.raises(ValueError):
            get_kt_shoulder_fillet(30, 20, 2, loading="shear")


class TestShaftGroove:
    """Test cases for Shaft.add_groove."""

    def test_add_groove_round_stores_row(self):
        """A round groove (a=0) derives width = 2 * radii and stores a Kt row."""
        shaft = Shaft(diameter=30.0, length=100.0)
        result = shaft.add_groove(position=50.0, diameter=20.0, radii=3.0)
        assert result is shaft
        number, dims, kt_axial, kt_bending, kt_torsion = shaft.grooves[0]
        assert number == 1
        assert dims == (50.0, 20.0, 3.0, 6.0)
        assert kt_axial == pytest.approx(get_kt_groove(30.0, 20.0, 3.0, loading="axial"))
        assert kt_bending == pytest.approx(get_kt_groove(30.0, 20.0, 3.0, loading="bending"))
        assert kt_torsion == pytest.approx(get_kt_groove(30.0, 20.0, 3.0, loading="torsion"))

    def test_add_groove_flat_stores_row(self):
        """A flat groove (a>0) uses a as the flat width and shoulder-fillet Kt."""
        shaft = Shaft(diameter=30.0, length=100.0)
        shaft.add_groove(position=50.0, diameter=20.0, radii=2.0, a=10.0)
        number, dims, kt_axial, kt_bending, kt_torsion = shaft.grooves[0]
        assert dims == (50.0, 20.0, 2.0, 10.0)
        expected_kt = get_kt_shoulder_fillet(30.0, 20.0, 2.0, loading="torsion")
        assert kt_torsion == pytest.approx(expected_kt)

    def test_add_groove_chains(self):
        """Repeated add_groove calls chain and number sequentially."""
        shaft = Shaft(diameter=30.0, length=100.0)
        shaft.add_groove(position=20.0, diameter=25.0, radii=1.0) \
            .add_groove(position=70.0, diameter=25.0, radii=1.0)
        assert [row[0] for row in shaft.grooves] == [1, 2]

    def test_add_groove_diameter_too_large_rejected(self):
        """A groove diameter not smaller than the shaft diameter is rejected."""
        shaft = Shaft(diameter=30.0, length=100.0)
        with pytest.raises(ValueError):
            shaft.add_groove(position=50.0, diameter=35.0, radii=1.0)

    def test_add_groove_non_positive_radii_rejected(self):
        """A non-positive radii is rejected."""
        shaft = Shaft(diameter=30.0, length=100.0)
        with pytest.raises(ValueError):
            shaft.add_groove(position=50.0, diameter=20.0, radii=0.0)

    def test_add_groove_negative_a_rejected(self):
        """A negative a is rejected."""
        shaft = Shaft(diameter=30.0, length=100.0)
        with pytest.raises(ValueError):
            shaft.add_groove(position=50.0, diameter=20.0, radii=1.0, a=-1.0)

    def test_add_groove_flat_radii_exceeding_half_a_rejected(self):
        """A flat groove's corner radius can't exceed half of a."""
        shaft = Shaft(diameter=30.0, length=100.0)
        with pytest.raises(ValueError):
            shaft.add_groove(position=50.0, diameter=20.0, radii=6.0, a=10.0)

    def test_add_groove_outside_shaft_rejected(self):
        """A groove spanning past the shaft's end is rejected."""
        shaft = Shaft(diameter=30.0, length=100.0)
        with pytest.raises(ValueError):
            shaft.add_groove(position=98.0, diameter=20.0, radii=3.0)


class TestShaftStressProfile:
    """Test cases for Shaft.stress_profile and Shaft.plot."""

    def test_no_peaks_without_grooves(self):
        """A plain shaft's profile has a constant nominal stress and no peaks."""
        shaft = Shaft(diameter=20.0, length=100.0)
        torque = 1.0e5
        profile = shaft.stress_profile(torque, n=10)
        assert profile["peaks"] == []
        expected = shaft.torsional_stress(torque)
        assert all(s == pytest.approx(expected) for s in profile["nominal_stress"])

    def test_groove_peak_matches_stored_kt(self):
        """The reported peak reuses the Kt row cached by add_groove."""
        shaft = Shaft(diameter=30.0, length=100.0)
        shaft.add_groove(position=50.0, diameter=20.0, radii=3.0)
        torque = 1.0e5
        profile = shaft.stress_profile(torque)
        assert len(profile["peaks"]) == 1
        peak = profile["peaks"][0]
        expected_kt = shaft.grooves[0][4]
        expected_nominal = Shaft(diameter=20.0, length=1.0).torsional_stress(torque)
        assert peak["x"] == pytest.approx(50.0)
        assert peak["kt"] == pytest.approx(expected_kt)
        assert peak["stress"] == pytest.approx(expected_kt * expected_nominal)

    def test_requires_at_least_two_samples(self):
        """n < 2 is rejected."""
        shaft = Shaft(diameter=20.0, length=100.0)
        with pytest.raises(ValueError):
            shaft.stress_profile(torque=1.0e5, n=1)

    def test_plot_returns_figure(self):
        """Smoke test for the matplotlib plot."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        shaft = Shaft(diameter=30.0, length=100.0)
        shaft.add_groove(position=50.0, diameter=20.0, radii=3.0, a=10.0)
        fig = shaft.plot(torque=1.0e5, show=False)
        assert isinstance(fig, Figure)


class TestSteppedShaft:
    """Test cases for the SteppedShaft class."""

    def _base(self, diameter=30.0, length=100.0):
        return Shaft(diameter=diameter, length=length)

    def test_creation_from_base_shaft(self):
        """A SteppedShaft starts as a single segment with no fillets."""
        ss = SteppedShaft(self._base())
        assert len(ss.segments) == 1
        assert ss.total_length == pytest.approx(100.0)
        assert ss.fillets == []

    def test_is_mecha_element(self):
        """SteppedShaft inherits from MechaElement."""
        assert isinstance(SteppedShaft(self._base()), MechaElement)

    def test_requires_shaft_instance(self):
        """A non-Shaft base is rejected."""
        with pytest.raises(ValueError):
            SteppedShaft("not a shaft")

    def test_add_shaft_right_appends(self):
        """side='right' (default) appends after the last segment."""
        ss = SteppedShaft(self._base(diameter=30.0, length=100.0))
        result = ss.add_shaft(Shaft(diameter=20.0, length=150.0), "fillet", 3.0)
        assert result is ss
        assert [s.diameter for s in ss.segments] == [30.0, 20.0]
        assert ss.total_length == pytest.approx(250.0)
        number, dims, kt_axial, kt_bending, kt_torsion = ss.fillets[0]
        assert dims == (30.0, 20.0, 3.0)
        expected_kt = get_kt_shoulder_fillet(30.0, 20.0, 3.0, loading="torsion")
        assert kt_torsion == pytest.approx(expected_kt)

    def test_add_shaft_left_prepends(self):
        """side='left' prepends before the first segment."""
        ss = SteppedShaft(self._base(diameter=30.0, length=100.0))
        ss.add_shaft(Shaft(diameter=40.0, length=80.0), "fillet", 4.0, side="left")
        assert [s.diameter for s in ss.segments] == [40.0, 30.0]
        assert ss.fillets[0][1] == (40.0, 30.0, 4.0)

    def test_add_shaft_chains_both_sides(self):
        """add_shaft can be chained, growing from either end."""
        ss = (
            SteppedShaft(self._base(diameter=30.0, length=100.0))
            .add_shaft(Shaft(diameter=20.0, length=150.0), "fillet", 3.0, side="right")
            .add_shaft(Shaft(diameter=25.0, length=80.0), "fillet", 2.0, side="left")
        )
        assert [s.diameter for s in ss.segments] == [25.0, 30.0, 20.0]
        assert len(ss.fillets) == 2

    def test_add_shaft_invalid_transition_rejected(self):
        """Only 'fillet' is a supported transition."""
        ss = SteppedShaft(self._base())
        with pytest.raises(ValueError):
            ss.add_shaft(Shaft(diameter=20.0, length=50.0), "chamfer", 3.0)

    def test_add_shaft_invalid_side_rejected(self):
        """side must be 'left' or 'right'."""
        ss = SteppedShaft(self._base())
        with pytest.raises(ValueError):
            ss.add_shaft(Shaft(diameter=20.0, length=50.0), "fillet", 3.0, side="up")

    def test_add_shaft_non_positive_radius_rejected(self):
        """A zero or negative fillet radius is rejected."""
        ss = SteppedShaft(self._base())
        with pytest.raises(ValueError):
            ss.add_shaft(Shaft(diameter=20.0, length=50.0), "fillet", 0.0)

    def test_add_shaft_radius_too_large_rejected(self):
        """A fillet radius that doesn't fit the smaller diameter is rejected."""
        ss = SteppedShaft(self._base(diameter=30.0))
        with pytest.raises(ValueError):
            # smaller diameter is 20mm
            ss.add_shaft(Shaft(diameter=20.0, length=50.0), "fillet", 15.0)

    def test_diameter_at_within_segments(self):
        """diameter_at reports each segment's own diameter."""
        ss = SteppedShaft(self._base(diameter=30.0, length=100.0))
        ss.add_shaft(Shaft(diameter=20.0, length=150.0), "fillet", 3.0)
        assert ss.diameter_at(50.0) == pytest.approx(30.0)
        assert ss.diameter_at(200.0) == pytest.approx(20.0)

    def test_diameter_at_out_of_range_rejected(self):
        """diameter_at rejects x outside [0, total_length]."""
        ss = SteppedShaft(self._base(diameter=30.0, length=100.0))
        ss.add_shaft(Shaft(diameter=20.0, length=150.0), "fillet", 3.0)
        with pytest.raises(ValueError):
            ss.diameter_at(-1.0)
        with pytest.raises(ValueError):
            ss.diameter_at(300.0)

    def test_diameter_at_accounts_for_segment_groove(self):
        """A groove added directly to a segment shows up via diameter_at."""
        base = self._base(diameter=30.0, length=100.0)
        base.add_groove(position=50.0, diameter=25.0, radii=2.0)
        ss = SteppedShaft(base)
        ss.add_shaft(Shaft(diameter=20.0, length=150.0), "fillet", 3.0)
        assert ss.diameter_at(50.0) == pytest.approx(25.0)
        assert ss.diameter_at(44.0) == pytest.approx(30.0)

    def test_stress_profile_includes_fillet_and_groove_peaks(self):
        """stress_profile reports both a fillet peak and a segment's groove peak."""
        base = self._base(diameter=30.0, length=100.0)
        base.add_groove(position=50.0, diameter=25.0, radii=2.0)
        ss = SteppedShaft(base)
        ss.add_shaft(Shaft(diameter=20.0, length=150.0), "fillet", 3.0)
        torque = 1.0e5
        profile = ss.stress_profile(torque)
        kinds = sorted(p["kind"] for p in profile["peaks"])
        assert kinds == ["fillet", "groove"]
        fillet_peak = next(p for p in profile["peaks"] if p["kind"] == "fillet")
        assert fillet_peak["x"] == pytest.approx(100.0)
        groove_peak = next(p for p in profile["peaks"] if p["kind"] == "groove")
        assert groove_peak["x"] == pytest.approx(50.0)

    def test_stress_profile_requires_at_least_two_samples(self):
        """n < 2 is rejected."""
        ss = SteppedShaft(self._base())
        ss.add_shaft(Shaft(diameter=20.0, length=50.0), "fillet", 3.0)
        with pytest.raises(ValueError):
            ss.stress_profile(torque=1.0e5, n=1)

    def test_stepped_shaft_repr(self):
        """Test stepped shaft string representation."""
        ss = SteppedShaft(self._base(diameter=30.0, length=100.0))
        ss.add_shaft(Shaft(diameter=20.0, length=150.0), "fillet", 3.0)
        assert "SteppedShaft" in repr(ss)
        assert "250" in repr(ss)


class TestSteppedShaftPlot:
    """Smoke test for the matplotlib plot."""

    def test_plot_returns_figure(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        base = Shaft(diameter=30.0, length=100.0)
        base.add_groove(position=50.0, diameter=25.0, radii=2.0, a=10.0)
        ss = SteppedShaft(base).add_shaft(Shaft(diameter=20.0, length=150.0), "fillet", 3.0)
        fig = ss.plot(torque=1.0e5, show=False)
        assert isinstance(fig, Figure)
