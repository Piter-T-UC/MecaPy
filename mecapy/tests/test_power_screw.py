"""Tests for the power screw module."""

import math

import pytest

from mecapy import MechaElement
from mecapy.shafts import PowerScrew
from mecapy.shafts.power_screw_data import get_thread_form


def shigley_screw():
    """Shigley Example 8-1 screw: square double thread, d=32, p=4."""
    return PowerScrew(major_diameter=32.0, pitch=4.0, length=200.0,
                      n_starts=2, thread_form="square")


class TestPowerScrewGeometry:
    """Geometry and construction."""

    def test_creation_and_inheritance(self):
        """Attributes are stored and the class inherits MechaElement."""
        screw = shigley_screw()
        assert screw.major_diameter == 32.0
        assert screw.pitch == 4.0
        assert screw.n_starts == 2
        assert screw.thread_form == "square"
        assert isinstance(screw, MechaElement)

    def test_geometry_properties(self):
        """Lead, mean/minor diameter and derived sections."""
        screw = shigley_screw()
        assert screw.lead == 8.0
        assert screw.mean_diameter == 30.0
        assert screw.minor_diameter == 28.0
        assert screw.lead_angle == pytest.approx(math.degrees(math.atan(8.0 / (math.pi * 30.0))))
        assert screw.root_area == pytest.approx(math.pi * 28.0 ** 2 / 4)
        assert screw.polar_moment == pytest.approx(math.pi * 28.0 ** 4 / 32)
        assert screw.second_moment == pytest.approx(math.pi * 28.0 ** 4 / 64)

    def test_thread_angle_per_form(self):
        """Named forms resolve to the tabulated half-angle."""
        common = dict(major_diameter=32.0, pitch=4.0, length=200.0)
        assert PowerScrew(thread_form="square", **common).thread_angle == 0.0
        assert PowerScrew(thread_form="acme", **common).thread_angle == 14.5
        assert PowerScrew(thread_form="trapezoidal", **common).thread_angle == 15.0
        assert PowerScrew(thread_form="buttress", **common).thread_angle == 7.0

    def test_custom_thread_angle(self):
        """Custom form stores the given half-angle; missing angle raises."""
        screw = PowerScrew(32.0, 4.0, 200.0, thread_form="custom", thread_angle=10.0)
        assert screw.thread_angle == 10.0
        with pytest.raises(ValueError):
            PowerScrew(32.0, 4.0, 200.0, thread_form="custom")
        with pytest.raises(ValueError):
            PowerScrew(32.0, 4.0, 200.0, thread_form="custom", thread_angle=95.0)

    def test_invalid_construction(self):
        """Non-physical inputs and unknown forms raise ValueError."""
        with pytest.raises(ValueError):
            PowerScrew(0.0, 4.0, 200.0)
        with pytest.raises(ValueError):
            PowerScrew(32.0, 0.0, 200.0)
        with pytest.raises(ValueError):
            PowerScrew(32.0, 4.0, 0.0)
        with pytest.raises(ValueError):
            PowerScrew(32.0, 4.0, 200.0, n_starts=0)
        with pytest.raises(ValueError):
            PowerScrew(32.0, 40.0, 200.0)  # pitch >= major diameter
        with pytest.raises(ValueError):
            PowerScrew(32.0, 4.0, 200.0, thread_form="bogus")

    def test_get_thread_form_invalid(self):
        """The data lookup raises for an unknown form."""
        with pytest.raises(ValueError):
            get_thread_form("bogus")


class TestPowerScrewTorque:
    """Torque, efficiency and self-locking (Shigley Example 8-1 anchors)."""

    def test_raise_torque(self):
        """Raising torque matches the Shigley Ex. 8-1 result (~26.2 N*m)."""
        screw = shigley_screw()
        t_r = screw.raise_torque(6400.0, mu=0.08, collar_diameter=40.0, mu_collar=0.08)
        assert t_r == pytest.approx(26176.0, rel=1e-3)

    def test_raise_torque_thread_only(self):
        """Without a collar only the thread term remains (~15.9 N*m)."""
        screw = shigley_screw()
        thread = screw.raise_torque(6400.0, mu=0.08)
        assert thread == pytest.approx(15937.0, rel=1e-3)

    def test_lower_torque(self):
        """Lowering torque: negative thread term (back-drives) plus the collar."""
        screw = shigley_screw()
        thread_only = screw.lower_torque(6400.0, mu=0.08)
        assert thread_only < 0  # not self-locking -> load back-drives
        t_l = screw.lower_torque(6400.0, mu=0.08, collar_diameter=40.0, mu_collar=0.08)
        assert t_l == pytest.approx(9774.4, rel=1e-3)

    def test_efficiency(self):
        """Overall efficiency (collar included) is about 0.31."""
        screw = shigley_screw()
        eff = screw.efficiency(6400.0, mu=0.08, collar_diameter=40.0, mu_collar=0.08)
        assert eff == pytest.approx(0.311, rel=1e-2)

    def test_self_locking(self):
        """Square thread with mu=0.08 back-drives (tan lambda = 0.0849 > 0.08)."""
        screw = shigley_screw()
        assert screw.is_self_locking(mu=0.08) is False
        assert screw.is_self_locking(mu=0.15) is True

    def test_thread_angle_increases_torque(self):
        """The sec(alpha) correction makes Acme need more torque than square."""
        common = dict(major_diameter=32.0, pitch=4.0, length=200.0, n_starts=2)
        square = PowerScrew(thread_form="square", **common)
        acme = PowerScrew(thread_form="acme", **common)
        assert acme.raise_torque(6400.0, mu=0.08) > square.raise_torque(6400.0, mu=0.08)

    def test_negative_friction_raises(self):
        """A negative friction coefficient is rejected."""
        screw = shigley_screw()
        with pytest.raises(ValueError):
            screw.raise_torque(6400.0, mu=-0.1)
        with pytest.raises(ValueError):
            screw.efficiency(0.0)


class TestPowerScrewStresses:
    """Body, thread-engagement and buckling stresses."""

    def test_axial_stress(self):
        """Axial stress is 4F/(pi*dr^2)."""
        screw = shigley_screw()
        assert screw.axial_stress(6400.0) == pytest.approx(4 * 6400.0 / (math.pi * 28.0 ** 2))

    def test_torsional_stress_matches_polar(self):
        """Torsional stress equals T*r/J on the root section."""
        screw = shigley_screw()
        torque = 26176.0
        expected = torque * 14.0 / (math.pi * 28.0 ** 4 / 32)
        assert screw.torsional_stress(torque) == pytest.approx(expected)

    def test_von_mises(self):
        """Von Mises combines axial and torsional stress."""
        screw = shigley_screw()
        f, t = 6400.0, 26176.0
        sigma = screw.axial_stress(f)
        tau = screw.torsional_stress(t)
        assert screw.von_mises_stress(f, t) == pytest.approx(math.sqrt(sigma ** 2 + 3 * tau ** 2))

    def test_thread_engagement_stresses(self):
        """Bearing, bending and shear use n_e = H/p and the right diameter."""
        screw = shigley_screw()
        f, h = 6400.0, 16.0  # 4 engaged threads
        n_e = h / 4.0
        assert screw.bearing_pressure(f, h) == pytest.approx(
            f / (math.pi * 30.0 * (0.5 * 4.0) * n_e)
        )
        assert screw.thread_bending_stress(f, h, side="screw") == pytest.approx(
            6 * f / (math.pi * 28.0 * n_e * 4.0)
        )
        assert screw.thread_bending_stress(f, h, side="nut") == pytest.approx(
            6 * f / (math.pi * 32.0 * n_e * 4.0)
        )
        assert screw.thread_shear_stress(f, h, side="screw") == pytest.approx(
            3 * f / (math.pi * 28.0 * n_e * 4.0)
        )

    def test_thread_stress_invalid(self):
        """Bad side and non-positive engagement length raise."""
        screw = shigley_screw()
        with pytest.raises(ValueError):
            screw.thread_bending_stress(6400.0, 16.0, side="middle")
        with pytest.raises(ValueError):
            screw.bearing_pressure(6400.0, 0.0)

    def test_buckling(self):
        """Euler critical load is pi^2*E*I/Le^2 and the check divides by force."""
        screw = shigley_screw()
        e_mpa = 210e9 / 1e6
        i = math.pi * 28.0 ** 4 / 64
        expected = math.pi ** 2 * e_mpa * i / 200.0 ** 2
        assert screw.critical_buckling_load(end_condition=1.0) == pytest.approx(expected)
        assert screw.check_buckling(50000.0) == pytest.approx(expected / 50000.0)
        with pytest.raises(ValueError):
            screw.check_buckling(0.0)

    def test_screw_safety_factor(self):
        """Safety factor is yield (MPa) over the von Mises stress."""
        screw = shigley_screw()
        f, t = 6400.0, 15937.0
        sy = 250e6 / 1e6
        expected = sy / screw.von_mises_stress(f, t)
        assert screw.screw_safety_factor(f, t) == pytest.approx(expected)

    def test_repr(self):
        """The repr names the class and key attributes."""
        screw = shigley_screw()
        assert "PowerScrew" in repr(screw)
        assert "square" in repr(screw)
