"""Tests for the Beam3D class and its CrossSection helper."""

import math

import pytest
from sympy import Symbol

from mecapy import MechaElement
from mecapy.beams import Beam3D, CrossSection
from mecapy.materials import add_custom_material

X = Symbol("x")


def steel_circular_beam(length=2.0, diameter=0.04):
    """A steel Beam3D on a solid circular section, no loads yet."""
    return Beam3D(length=length, section=CrossSection.circular(diameter))


class TestCrossSection:
    """Test cases for the CrossSection class."""

    def test_circular_values(self):
        """Circular section matches the hand-computed A, I, J and c."""
        d = 0.04
        section = CrossSection.circular(d)
        assert section.area == pytest.approx(math.pi * d ** 2 / 4)
        assert section.second_moment_y == pytest.approx(math.pi * d ** 4 / 64)
        assert section.second_moment_z == pytest.approx(math.pi * d ** 4 / 64)
        assert section.polar_moment == pytest.approx(math.pi * d ** 4 / 32)
        assert section.c_y == pytest.approx(d / 2)
        assert section.c_z == pytest.approx(d / 2)

    def test_rectangular_values(self):
        """Rectangular section matches the hand-computed A, I and c."""
        b, h = 0.03, 0.05
        section = CrossSection.rectangular(width=b, height=h)
        assert section.area == pytest.approx(b * h)
        assert section.second_moment_z == pytest.approx(b * h ** 3 / 12)
        assert section.second_moment_y == pytest.approx(h * b ** 3 / 12)
        assert section.c_y == pytest.approx(h / 2)
        assert section.c_z == pytest.approx(b / 2)
        assert section.polar_moment > 0

    def test_invalid_constructor_inputs(self):
        """Every non-positive constructor input raises ValueError."""
        good = dict(area=1e-3, second_moment_y=1e-7, second_moment_z=1e-7,
                    c_y=0.02, c_z=0.02)
        for key in good:
            bad = dict(good, **{key: 0})
            with pytest.raises(ValueError):
                CrossSection(**bad)

    def test_polar_moment_defaults_to_sum(self):
        """J defaults to I_y + I_z (perpendicular-axis theorem)."""
        section = CrossSection(area=1e-3, second_moment_y=1e-7,
                               second_moment_z=3e-7, c_y=0.02, c_z=0.02)
        assert section.polar_moment == pytest.approx(4e-7)

    def test_non_numeric_input(self):
        """A non-numeric input raises ValueError."""
        with pytest.raises(ValueError):
            CrossSection.circular("wide")

    def test_invalid_circular(self):
        """A non-positive diameter raises ValueError."""
        with pytest.raises(ValueError):
            CrossSection.circular(0)

    def test_invalid_rectangular(self):
        """A non-positive side raises ValueError."""
        with pytest.raises(ValueError):
            CrossSection.rectangular(-1, 2)

    def test_setter_revalidates(self):
        """Setting an attribute after construction re-runs validation."""
        section = CrossSection.circular(0.04)
        with pytest.raises(ValueError):
            section.area = -1

    def test_repr(self):
        """Repr names the class and shows the area."""
        section = CrossSection.circular(0.04)
        assert "CrossSection" in repr(section)


class TestBeam3DConstruction:
    """Test cases for Beam3D construction and validation."""

    def test_moduli_default_from_material(self):
        """E and G default to the steel database values."""
        beam = steel_circular_beam()
        assert beam.elastic_modulus == pytest.approx(210e9)
        assert beam.shear_modulus == pytest.approx(81e9)

    def test_explicit_moduli(self):
        """Explicit E and G override the material values."""
        beam = Beam3D(length=1.0, section=CrossSection.circular(0.02),
                      elastic_modulus=200e9, shear_modulus=77e9)
        assert beam.elastic_modulus == pytest.approx(200e9)
        assert beam.shear_modulus == pytest.approx(77e9)

    def test_shear_modulus_poisson_fallback(self):
        """G falls back to E/(2*(1+nu)) when only poisson_ratio exists."""
        add_custom_material("b3d_poisson_only", {
            "elastic_modulus": 2.6e9, "poisson_ratio": 0.3, "yield_strength": 60e6,
        })
        beam = Beam3D(length=1.0, section=CrossSection.circular(0.02),
                      material="b3d_poisson_only")
        assert beam.shear_modulus == pytest.approx(2.6e9 / 2.6)

    def test_shear_modulus_unresolvable(self):
        """A material with neither G nor poisson_ratio raises ValueError."""
        add_custom_material("b3d_e_only", {
            "elastic_modulus": 1e9, "yield_strength": 30e6,
        })
        with pytest.raises(ValueError):
            Beam3D(length=1.0, section=CrossSection.circular(0.02), material="b3d_e_only")

    def test_invalid_length(self):
        """A non-positive length raises ValueError."""
        with pytest.raises(ValueError):
            Beam3D(length=0, section=CrossSection.circular(0.02))

    def test_invalid_section(self):
        """A section that is not a CrossSection raises ValueError."""
        with pytest.raises(ValueError):
            Beam3D(length=1.0, section=0.04)

    def test_unknown_material(self):
        """An unknown material name raises ValueError."""
        with pytest.raises(ValueError):
            Beam3D(length=1.0, section=CrossSection.circular(0.02), material="unobtanium")

    def test_is_mecha_element(self):
        """Beam3D inherits from MechaElement."""
        assert isinstance(steel_circular_beam(), MechaElement)

    def test_repr(self):
        """Repr names the class and shows the length."""
        beam = steel_circular_beam(length=2.0)
        assert "Beam3D" in repr(beam)
        assert "2.0" in repr(beam)


class TestBeam3DBuilders:
    """Test cases for the load/support builder methods."""

    def test_chaining(self):
        """Builder methods return self so calls can be chained."""
        beam = steel_circular_beam()
        result = (beam.add_support(0, "pin").add_support(2.0, "roller")
                  .add_point_load(-100, 1.0).add_torque(50, 1.5))
        assert result is beam

    def test_support_out_of_range(self):
        """A support outside [0, length] raises ValueError."""
        with pytest.raises(ValueError):
            steel_circular_beam().add_support(3.0)

    def test_bad_support_kind(self):
        """An unrecognized support kind raises ValueError."""
        with pytest.raises(ValueError):
            steel_circular_beam().add_support(0, "clamped")

    def test_load_out_of_range(self):
        """A load outside [0, length] raises ValueError."""
        with pytest.raises(ValueError):
            steel_circular_beam().add_point_load(-100, 2.5)

    def test_bad_load_dir(self):
        """An invalid load direction raises ValueError."""
        with pytest.raises(ValueError):
            steel_circular_beam().add_point_load(-100, 1.0, dir="w")

    def test_bad_load_order(self):
        """Order below -1 raises ValueError (moments have own methods)."""
        with pytest.raises(ValueError):
            steel_circular_beam().add_load(-100, 1.0, -2)

    def test_axial_order_restricted(self):
        """Axial loads only accept orders -1 and 0."""
        with pytest.raises(ValueError):
            steel_circular_beam().add_load(-100, 1.0, 1, dir="x")

    def test_bad_distributed_end(self):
        """A distributed end before its start raises ValueError."""
        with pytest.raises(ValueError):
            steel_circular_beam().add_distributed_load(-100, 1.5, 1.0)

    def test_torque_out_of_range(self):
        """A torque outside [0, length] raises ValueError."""
        with pytest.raises(ValueError):
            steel_circular_beam().add_torque(50, -0.5)

    def test_moment_dir_x_is_torque(self):
        """add_moment with dir='x' registers a torque."""
        beam = steel_circular_beam().add_moment(50, 1.0, dir="x")
        assert float(beam.torsional_moment().subs(X, 0.5)) == pytest.approx(50)


class TestBeam3DResults:
    """Analytic cross-checks of the solved results."""

    def test_simply_supported_reactions(self):
        """A central y load splits evenly; the z plane stays unloaded."""
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller").add_point_load(-1000, 1.0)
        reactions = beam.reactions
        assert sorted(float(v) for v in reactions["y"].values()) == pytest.approx([500, 500])
        assert all(float(v) == pytest.approx(0) for v in reactions["z"].values())

    def test_simply_supported_shear(self):
        """Shear is +P/2 before and -P/2 after a central load."""
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller").add_point_load(-1000, 1.0)
        shear = beam.shear_force()["y"]
        assert abs(float(shear.subs(X, 0.5))) == pytest.approx(500)
        assert abs(float(shear.subs(X, 1.5))) == pytest.approx(500)

    def test_simply_supported_max_moment(self):
        """Max bending moment of a central load is P*L/4 at midspan."""
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller").add_point_load(-1000, 1.0)
        location, moment = beam.max_bending_moment()["y"]
        assert float(location) == pytest.approx(1.0)
        assert float(moment) == pytest.approx(500)  # 1000 * 2 / 4

    def test_simply_supported_deflection(self):
        """Central-load midspan deflection integrates to P*L^3/(48*E*I)."""
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller").add_point_load(-1000, 1.0)
        expected = 1000 * 2.0 ** 3 / (48 * 210e9 * beam.section.second_moment_z)
        deflection = float(beam.deflection()["y"].subs(X, 1.0))
        assert deflection == pytest.approx(-expected)

    def test_uniform_load_deflection(self):
        """Uniform-load midspan deflection is 5*q*L^4/(384*E*I), z plane."""
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller")
        beam.add_distributed_load(-500, 0, 2.0, dir="z")
        expected = 5 * 500 * 2.0 ** 4 / (384 * 210e9 * beam.section.second_moment_y)
        deflection = float(beam.deflection()["z"].subs(X, 1.0))
        assert deflection == pytest.approx(-expected)

    def test_cantilever(self):
        """Cantilever tip deflection is P*L^3/(3*E*I) with reacting moment."""
        beam = steel_circular_beam()
        beam.add_support(0, "fixed").add_point_load(-1000, 2.0)
        reactions = beam.reactions["y"]
        values = sorted(float(v) for v in reactions.values())
        assert values == pytest.approx([-2000, 1000])  # M_0 = -P*L, R_0 = P
        expected = 1000 * 2.0 ** 3 / (3 * 210e9 * beam.section.second_moment_z)
        assert float(beam.deflection()["y"].subs(X, 2.0)) == pytest.approx(-expected)

    def test_planes_are_independent(self):
        """Simultaneous y and z loads solve each plane independently."""
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller")
        beam.add_point_load(-1000, 1.0, dir="y").add_point_load(-600, 0.5, dir="z")
        reactions = beam.reactions
        assert sum(float(v) for v in reactions["y"].values()) == pytest.approx(1000)
        assert sum(float(v) for v in reactions["z"].values()) == pytest.approx(600)

    def test_float_positions_solve(self):
        """Float support/load positions solve without sympy's IndexError."""
        beam = Beam3D(length=1.0, section=CrossSection.circular(0.02))
        beam.add_support(0.35, "pin").add_support(1.0, "roller")
        beam.add_point_load(-1000, 0.7)
        reactions = beam.reactions["y"]
        assert sum(float(v) for v in reactions.values()) == pytest.approx(1000)

    def test_axial_force_profile(self):
        """Self-balanced axial loads give a compressive step between them."""
        beam = steel_circular_beam()
        beam.add_point_load(1000, 0.5, dir="x").add_point_load(-1000, 1.5, dir="x")
        axial = beam.axial_force()
        assert float(axial.subs(X, 0.2)) == pytest.approx(0)
        assert float(axial.subs(X, 1.0)) == pytest.approx(-1000)
        assert float(axial.subs(X, 1.8)) == pytest.approx(0)

    def test_axial_stress(self):
        """Axial stress is N(x)/A."""
        beam = steel_circular_beam()
        beam.add_point_load(2000, 2.0, dir="x")  # pulled at the tip, restrained at 0
        stress = beam.axial_stress()
        assert float(stress.subs(X, 1.0)) == pytest.approx(2000 / beam.section.area)

    def test_torsion(self):
        """A tip torque gives constant T, twist T*L/(G*J) and tau=T*c/J."""
        beam = steel_circular_beam()
        beam.add_torque(100, 2.0)
        section = beam.section
        assert float(beam.torsional_moment().subs(X, 1.0)) == pytest.approx(100)
        expected_twist = 100 * 2.0 / (81e9 * section.polar_moment)
        assert float(beam.twist().subs(X, 2.0)) == pytest.approx(expected_twist)
        expected_stress = 100 * 0.02 / section.polar_moment
        assert float(beam.torsional_stress().subs(X, 1.0)) == pytest.approx(expected_stress)

    def test_bending_stress(self):
        """Midspan bending stress of a central load is (P*L/4)*c/I —
        negative at the +c fiber (compression on top, sympy's sagging
        sign convention)."""
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller").add_point_load(-1000, 1.0)
        section = beam.section
        expected = 500 * section.c_y / section.second_moment_z
        stress = float(beam.bending_stress("y").subs(X, 1.0))
        assert stress == pytest.approx(-expected)

    def test_bending_stress_bad_dir(self):
        """An invalid bending_stress direction raises ValueError."""
        with pytest.raises(ValueError):
            steel_circular_beam().bending_stress("w")

    def test_function_load_constant_matches_distributed(self):
        """A constant q = f(x) reproduces add_distributed_load exactly."""
        reference = steel_circular_beam()
        reference.add_support(0, "pin").add_support(2.0, "roller")
        reference.add_distributed_load(-500, 0, 2.0)
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller")
        beam.add_function_load(lambda x: -500, segments=8)
        expected = float(reference.deflection()["y"].subs(X, 1.0))
        assert float(beam.deflection()["y"].subs(X, 1.0)) == pytest.approx(expected)

    def test_function_load_triangular_reactions(self):
        """A triangular q(x) = -w*x gives reactions total/3 and 2*total/3."""
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller")
        beam.add_function_load(lambda x: -300 * x, segments=100)  # total 600 N
        reactions = sorted(float(v) for v in beam.reactions["y"].values())
        assert reactions[0] == pytest.approx(200, rel=1e-3)
        assert reactions[1] == pytest.approx(400, rel=1e-3)

    def test_function_load_validation(self):
        """Bad function-load inputs raise ValueError."""
        beam = steel_circular_beam()
        with pytest.raises(ValueError):
            beam.add_function_load(123)
        with pytest.raises(ValueError):
            beam.add_function_load(lambda x: -1, start=1.5, end=1.0)
        with pytest.raises(ValueError):
            beam.add_function_load(lambda x: -1, segments=0)
        with pytest.raises(ValueError):
            beam.add_function_load(lambda x: -1, dir="w")


class TestBeam3DRecompute:
    """The recompute-don't-cache guarantee across input changes."""

    def loaded_beam(self):
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller").add_point_load(-1000, 1.0)
        return beam

    def test_section_swap_updates_deflection(self):
        """Swapping in a section with doubled I halves the deflection."""
        beam = self.loaded_beam()
        before = float(beam.deflection()["y"].subs(X, 1.0))
        old = beam.section
        beam.section = CrossSection(
            area=old.area, second_moment_y=2 * old.second_moment_y,
            second_moment_z=2 * old.second_moment_z,
            c_y=old.c_y, c_z=old.c_z,
        )
        after = float(beam.deflection()["y"].subs(X, 1.0))
        assert after == pytest.approx(before / 2)

    def test_section_mutation_updates_stress(self):
        """Mutating the section's area in place changes axial stress."""
        beam = steel_circular_beam().add_point_load(1000, 2.0, dir="x")
        before = float(beam.axial_stress().subs(X, 1.0))
        beam.section.area = beam.section.area * 2
        after = float(beam.axial_stress().subs(X, 1.0))
        assert after == pytest.approx(before / 2)

    def test_length_setter_revalidates(self):
        """Setting a non-positive length after construction raises."""
        beam = steel_circular_beam()
        with pytest.raises(ValueError):
            beam.length = -1


class TestBeam3DPlots:
    """Smoke tests: every plot method returns a matplotlib Figure."""

    @pytest.fixture(autouse=True)
    def _matplotlib(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

    def full_beam(self):
        """A beam exercising every load type for the plot methods."""
        beam = steel_circular_beam()
        beam.add_support(0, "pin").add_support(2.0, "roller")
        beam.add_point_load(-1000, 1.0, dir="y").add_point_load(-400, 1.5, dir="z")
        beam.add_distributed_load(-200, 0.2, 0.8, dir="y")
        beam.add_point_load(500, 2.0, dir="x")
        beam.add_moment(150, 0.5, dir="z")
        beam.add_torque(80, 2.0)
        return beam

    @pytest.mark.parametrize("method", [
        "plot_shear_force", "plot_bending_moment", "plot_deflection",
        "plot_bending_stress", "plot_axial_stress", "plot_torsional_stress",
        "plot_results", "draw",
    ])
    def test_plot_returns_figure(self, method):
        """Each plot method returns a Figure without showing it."""
        from matplotlib.figure import Figure
        fig = getattr(self.full_beam(), method)(show=False)
        assert isinstance(fig, Figure)

    def test_single_direction_plot(self):
        """dir='y' plots a single plane and still returns a Figure."""
        from matplotlib.figure import Figure
        fig = self.full_beam().plot_shear_force(dir="y", show=False)
        assert isinstance(fig, Figure)

    def test_plot_bad_dir(self):
        """An invalid plot direction raises ValueError."""
        with pytest.raises(ValueError):
            self.full_beam().plot_shear_force(dir="w", show=False)
