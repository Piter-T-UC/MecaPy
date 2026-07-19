"""3D beam analysis: bending in two planes, axial force and torsion.

Units convention: consistent SI — lengths in m, forces in N, moments and
torques in N*m, stresses in Pa — matching the beams module (Pa, m).

Coordinate convention: x is the beam axis; y and z are the transverse
directions. Loads in y bend about z (plane "y", uses I_z); loads in z
bend about y (plane "z", uses I_y) — the same convention as
:class:`~mecapy.shafts.Shaft`.
"""

import math

from sympy import Rational, SingularityFunction, Symbol, integrate, sympify

from ..base import MechaElement
from .beam import Beam
from .section import CrossSection

_X = Symbol("x")  # mecapy.beams.Beam's own sympy variable, matched here for substitution

_DIRS = ("x", "y", "z")
_PLANES = ("y", "z")

# Plot palette shared with the rest of mecapy (see Shaft.plot).
_COLOR_Y = "#3c25eb"
_COLOR_Z = "#1f2937"
_COLOR_GREY = "#9ca3af"
_COLOR_GRID = "#e5e7eb"
_COLOR_ALERT = "#d90b0b"


def _pyplot(feature):
    """Lazily import matplotlib.pyplot, raising a helpful error if absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            f"matplotlib is required for {feature}; install it with 'pip install matplotlib'"
        )
    return plt


class Beam3D(MechaElement):
    """
    Beam analysis with two bending planes, axial loading and torsion.

    The public API is modeled on
    ``sympy.physics.continuum_mechanics.beam.Beam3D`` (direction-aware
    loads via ``dir="x"/"y"/"z"``), but the solving engine is different:
    sympy's ``Beam3D`` excludes point loads from its shear, moment and
    deflection results (it assumes span-wide distributed loads), so this
    class instead solves each transverse plane with a 2D
    :class:`~mecapy.beams.Beam` — full singularity-function integration
    that handles point loads, moments, distributed loads and pin/roller/
    fixed supports correctly. Axial force and torque diagrams are direct
    singularity-function sums.

    Deflection is Euler-Bernoulli (no Timoshenko shear deformation),
    which is accurate for slender members. For combined von Mises stress
    on circular members see :meth:`mecapy.shafts.Shaft.stress_profile`.

    Every result recomputes from the stored supports/loads on access, so
    changing ``length``, ``section`` (or mutating the section in place)
    or adding loads automatically updates all results.

    Attributes:
        length (float): Beam length in m. Settable.
        section (CrossSection): Cross-section geometry. Settable.
        elastic_modulus (float): Young's modulus E in Pa. Settable.
        shear_modulus (float): Shear modulus G in Pa. Settable.
        material (str): Material type.
    """

    def __init__(self, length, section, material="steel",
                 elastic_modulus=None, shear_modulus=None, name=None):
        """
        Initialize a Beam3D.

        Args:
            length (float): Beam length in m. Must be strictly positive.
            section (CrossSection): Cross-section geometry.
            material (str): Material name (default: "steel").
            elastic_modulus (float): Young's modulus E in Pa. If ``None``,
                the material database value is used. Must be numeric —
                unlike the 2D :class:`Beam` there is no symbolic mode, so
                plotting always works.
            shear_modulus (float): Shear modulus G in Pa. If ``None``, the
                material's ``shear_modulus`` is used, falling back to
                ``E / (2 * (1 + poisson_ratio))`` if only a Poisson ratio
                is available.
            name (str): Optional identifier for the beam.

        Raises:
            ValueError: If ``length`` is not strictly positive, ``section``
                is not a :class:`CrossSection`, a modulus is not strictly
                positive, or no shear modulus can be determined.
        """
        super().__init__(name=name, material=material)
        self.length = length
        self.section = section
        if elastic_modulus is None:
            elastic_modulus = self.material_properties["elastic_modulus"]
        self.elastic_modulus = elastic_modulus
        if shear_modulus is None:
            props = self.material_properties
            if "shear_modulus" in props:
                shear_modulus = props["shear_modulus"]
            elif "poisson_ratio" in props:
                shear_modulus = self.elastic_modulus / (2 * (1 + props["poisson_ratio"]))
            else:
                raise ValueError(
                    f"shear_modulus is required: material {self.material!r} has neither "
                    "'shear_modulus' nor 'poisson_ratio'"
                )
        self.shear_modulus = shear_modulus
        self._supports = []  # (location, kind)
        self._loads = []  # (value, start, order, dir, end) — forces only
        self._moments = []  # (value, location, dir) — dir is the axis the moment acts about
        self._torques = []  # (value, location)

    # ---- Settable primary inputs ----

    @property
    def length(self):
        """float: Beam length in m."""
        return self._length

    @length.setter
    def length(self, value):
        if value <= 0:
            raise ValueError("Beam length must be strictly positive")
        self._length = value

    @property
    def section(self):
        """CrossSection: Cross-section geometry (held by reference)."""
        return self._section

    @section.setter
    def section(self, value):
        if not isinstance(value, CrossSection):
            raise ValueError(f"section must be a CrossSection instance, got {value!r}")
        self._section = value

    @property
    def elastic_modulus(self):
        """float: Young's modulus E in Pa."""
        return self._elastic_modulus

    @elastic_modulus.setter
    def elastic_modulus(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"elastic_modulus must be a number, got {value!r}")
        if value <= 0:
            raise ValueError("elastic_modulus must be strictly positive")
        self._elastic_modulus = value

    @property
    def shear_modulus(self):
        """float: Shear modulus G in Pa."""
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"shear_modulus must be a number, got {value!r}")
        if value <= 0:
            raise ValueError("shear_modulus must be strictly positive")
        self._shear_modulus = value

    # ---- Model building (all return self for chaining) ----

    def add_support(self, location, kind="pin"):
        """
        Add a support at an axial location, restraining deflection in
        both transverse planes (like a bearing).

        Args:
            location (float): Axial position in m.
            kind (str): "pin", "roller", or "fixed" (default: "pin").

        Returns:
            Beam3D: self, so calls can be chained.

        Raises:
            ValueError: If location is outside [0, length] or kind is
                unrecognized.
        """
        if not 0 <= location <= self.length:
            raise ValueError(f"Support at location={location}m is outside the beam's length")
        if kind not in ("pin", "roller", "fixed"):
            raise ValueError("kind must be 'pin', 'roller', or 'fixed'")
        self._supports.append((location, kind))
        return self

    def add_load(self, value, start, order, dir="y", end=None):
        """
        Add a force load using sympy's singularity-function convention.

        Args:
            value (float): Load magnitude — N for order -1, N/m for
                order 0, and so on. Negative acts opposite the axis.
            start (float): Application start position in m.
            order (int): Singularity order (-1 point load, 0 uniform,
                1 ramp, ...). Concentrated moments have their own methods
                (:meth:`add_moment` / :meth:`add_torque`).
            dir (str): Load direction — "x" (axial), "y" or "z"
                (default: "y"). Axial loads accept orders -1 and 0 only.
            end (float): Optional end position in m for distributed loads
                (order >= 0).

        Returns:
            Beam3D: self, so calls can be chained.

        Raises:
            ValueError: If dir or order is invalid, or a position is
                outside [0, length].
        """
        if dir not in _DIRS:
            raise ValueError("dir must be 'x', 'y', or 'z'")
        if order != int(order) or order < -1:
            raise ValueError(
                "order must be an integer >= -1; use add_moment/add_torque for "
                "concentrated moments"
            )
        order = int(order)
        if dir == "x" and order not in (-1, 0):
            raise ValueError("axial loads (dir='x') accept orders -1 and 0 only")
        if not 0 <= start <= self.length:
            raise ValueError(f"Load at start={start}m is outside the beam's length")
        if end is not None:
            if order < 0:
                raise ValueError("end only applies to distributed loads (order >= 0)")
            if not start < end <= self.length:
                raise ValueError(f"Load end={end}m must lie in ({start}, {self.length}]")
        self._loads.append((value, start, order, dir, end))
        return self

    def add_point_load(self, value, location, dir="y"):
        """Add a concentrated force (order -1) in a direction, in N."""
        return self.add_load(value, location, -1, dir=dir)

    def add_distributed_load(self, value, start, end, dir="y"):
        """Add a uniformly distributed force (order 0) in N/m between two points."""
        return self.add_load(value, start, 0, dir=dir, end=end)

    def add_function_load(self, func, start=0, end=None, segments=50, dir="y"):
        """
        Add a distributed load given as an arbitrary function q = f(x)
        (N/m), discretized into ``segments`` piecewise-constant
        distributed loads — each segment carries the value of ``f`` at
        its midpoint. The approximation improves with more segments; 50
        is ample for smooth load shapes.

        Args:
            func (callable): Load intensity ``f(x)`` in N/m — takes a
                position in m (float) and returns N/m (negative acts
                opposite the axis).
            start (float): Start of the loaded span in m (default: 0).
            end (float): End of the loaded span in m (default: the
                beam's full length).
            segments (int): Number of constant-load segments (>= 1,
                default: 50).
            dir (str): Load direction — "x" (axial), "y" or "z"
                (default: "y").

        Returns:
            Beam3D: self, so calls can be chained.

        Raises:
            ValueError: If ``func`` is not callable, ``segments`` is not
                a positive integer, dir is invalid, or the span is not
                within ``[0, length]``.
        """
        if not callable(func):
            raise ValueError("func must be callable, q = f(x)")
        if segments != int(segments) or segments < 1:
            raise ValueError("segments must be a positive integer")
        if end is None:
            end = self.length
        if not 0 <= start < end <= self.length:
            raise ValueError(
                f"Load span [{start}, {end}] must lie within [0, {self.length}]"
            )
        segments = int(segments)
        width = (end - start) / segments
        for i in range(segments):
            x0 = start + i * width
            # last segment ends exactly at `end` — float accumulation can
            # otherwise overshoot the beam length by one ulp
            x1 = end if i == segments - 1 else x0 + width
            value = float(func((x0 + x1) / 2))
            if value:
                self.add_load(value, x0, 0, dir=dir, end=x1)
        return self

    def add_moment(self, value, location, dir="y"):
        """
        Add a concentrated moment about an axis.

        A moment about z bends the beam in plane "y" and vice versa;
        ``dir="x"`` is a torque and delegates to :meth:`add_torque`
        (mirroring sympy Beam3D's ``apply_moment_load(dir="x")``).

        Args:
            value (float): Moment magnitude in N*m.
            location (float): Axial position in m.
            dir (str): Axis the moment acts about — "x", "y" or "z"
                (default: "y").

        Returns:
            Beam3D: self, so calls can be chained.

        Raises:
            ValueError: If dir is invalid or location is outside
                [0, length].
        """
        if dir not in _DIRS:
            raise ValueError("dir must be 'x', 'y', or 'z'")
        if dir == "x":
            return self.add_torque(value, location)
        if not 0 <= location <= self.length:
            raise ValueError(f"Moment at location={location}m is outside the beam's length")
        self._moments.append((value, location, dir))
        return self

    def add_torque(self, value, location):
        """
        Add a concentrated torque about the beam axis.

        Args:
            value (float): Torque magnitude in N*m.
            location (float): Axial position in m.

        Returns:
            Beam3D: self, so calls can be chained.

        Raises:
            ValueError: If location is outside [0, length].
        """
        if not 0 <= location <= self.length:
            raise ValueError(f"Torque at location={location}m is outside the beam's length")
        self._torques.append((value, location))
        return self

    # ---- Internal engine ----

    def _has_plane_loads(self, plane):
        """Whether any stored load/moment acts in a bending plane."""
        force_dir = plane
        moment_dir = "z" if plane == "y" else "y"
        return any(load[3] == force_dir for load in self._loads) or any(
            moment[2] == moment_dir for moment in self._moments
        )

    def _build_beam(self, plane):
        """Build a fresh 2D Beam for one bending plane from the stored
        supports/loads — the only place that touches Beam's API.

        Positions are converted to exact sympy.Rational before reaching
        Beam/sympy: sympy's solve_for_reaction_loads silently breaks (an
        IndexError deep inside linsolve) when the beam length or a
        support/load location is a plain Python float rather than an
        exact number. Magnitudes don't need this; only positions do.
        """
        force_dir = plane
        moment_dir = "z" if plane == "y" else "y"
        inertia = (
            self.section.second_moment_z if plane == "y" else self.section.second_moment_y
        )
        beam = Beam(
            Rational(self.length),
            material=self.material,
            elastic_modulus=self.elastic_modulus,
            second_moment=inertia,
        )
        for location, kind in self._supports:
            beam.add_support(Rational(location), kind)
        for value, start, order, dir_, end in self._loads:
            if dir_ != force_dir:
                continue
            beam.add_load(
                value, Rational(start), order,
                end=None if end is None else Rational(end),
            )
        for value, location, dir_ in self._moments:
            if dir_ != moment_dir:
                continue
            beam.add_moment(value, Rational(location))
        return beam

    def _axial_force_expr(self):
        """Internal axial force N(x) as a singularity-function sum.

        Equilibrium of the segment right of the cut: N(x) = sum of every
        applied axial load at or after x, with the axial restraint
        (thrust support) at x = 0 reacting any imbalance. Positive =
        tension. Statically indeterminate axial cases (fixed-fixed) are
        out of scope.
        """
        expr = sympify(0)
        for value, start, order, dir_, end in self._loads:
            if dir_ != "x":
                continue
            if order == -1:
                expr += value * (1 - SingularityFunction(_X, Rational(start), 0))
            else:  # order == 0 (validated in add_load)
                stop = self.length if end is None else end
                expr += value * (
                    (stop - start)
                    - SingularityFunction(_X, Rational(start), 1)
                    + SingularityFunction(_X, Rational(stop), 1)
                )
        return expr

    def _torque_expr(self):
        """Internal torque T(x): sum of torques applied at or after x,
        with the torsional restraint at x = 0 reacting any imbalance.
        Statically indeterminate torsion (fixed-fixed) is out of scope.
        """
        expr = sympify(0)
        for value, location in self._torques:
            expr += value * (1 - SingularityFunction(_X, Rational(location), 0))
        return expr

    # ---- Results (all recompute from the stored model on access) ----

    @property
    def reactions(self):
        """dict: Solved reaction loads per bending plane,
        ``{"y": {...}, "z": {...}}``."""
        return {plane: self._build_beam(plane).reactions for plane in _PLANES}

    def shear_force(self):
        """
        Shear force diagrams in both bending planes.

        Returns:
            dict: ``{"y": Vy(x), "z": Vz(x)}`` in N, symbolic expressions
            in ``x`` — substitute ``x`` for a numeric value.
        """
        return {plane: self._build_beam(plane).shear_force() for plane in _PLANES}

    def bending_moment(self):
        """
        Bending moment diagrams in both planes.

        Returns:
            dict: ``{"y": M(x), "z": M(x)}`` in N*m — "y" is the moment
            produced by y-direction forces and z-moments (bending about
            z), "z" by z-direction forces and y-moments (bending about
            y). Symbolic expressions in ``x``.
        """
        return {plane: self._build_beam(plane).bending_moment() for plane in _PLANES}

    def slope(self):
        """
        Slope diagrams in both planes.

        Returns:
            dict: ``{"y": expr, "z": expr}`` in rad, symbolic in ``x``.
        """
        return {plane: self._build_beam(plane).slope() for plane in _PLANES}

    def deflection(self):
        """
        Deflection diagrams in both planes, obtained by sympy's
        singularity-function integration with boundary conditions (not
        closed-form table formulas).

        Returns:
            dict: ``{"y": expr, "z": expr}`` in m, symbolic in ``x``.
        """
        return {plane: self._build_beam(plane).deflection() for plane in _PLANES}

    def max_bending_moment(self):
        """
        Maximum bending moment in each plane.

        Returns:
            dict: ``{"y": (location, value), "z": (location, value)}`` in
            (m, N*m); ``(0.0, 0.0)`` for a plane with no loads.
        """
        return {
            plane: (
                self._build_beam(plane).max_bending_moment()
                if self._has_plane_loads(plane)
                else (0.0, 0.0)
            )
            for plane in _PLANES
        }

    def max_deflection(self):
        """
        Maximum deflection in each plane.

        Returns:
            dict: ``{"y": (location, value), "z": (location, value)}`` in
            (m, m); ``(0.0, 0.0)`` for a plane with no loads.
        """
        return {
            plane: (
                self._build_beam(plane).max_deflection()
                if self._has_plane_loads(plane)
                else (0.0, 0.0)
            )
            for plane in _PLANES
        }

    def axial_force(self):
        """
        Internal axial force diagram N(x).

        Returns:
            Axial force in N, symbolic in ``x`` (see
            :meth:`_axial_force_expr` for the reaction convention).
        """
        return self._axial_force_expr()

    def torsional_moment(self):
        """
        Internal torque diagram T(x).

        Returns:
            Torque in N*m, symbolic in ``x`` (see :meth:`_torque_expr`
            for the reaction convention).
        """
        return self._torque_expr()

    def twist(self):
        """
        Angular twist phi(x) = integral of T / (G*J), with phi(0) = 0.

        Returns:
            Twist angle in rad, symbolic in ``x``.
        """
        return integrate(self._torque_expr(), _X) / (
            self.shear_modulus * self.section.polar_moment
        )

    # ---- Stresses (expressions in x, Pa) ----

    def axial_stress(self):
        """
        Axial stress sigma = N(x) / A. Positive tension.

        Returns:
            Axial stress in Pa, symbolic in ``x``.
        """
        return self._axial_force_expr() / self.section.area

    def torsional_stress(self):
        """
        Torsional shear stress tau = T(x) * c / J at the outer fiber,
        with ``c = max(c_y, c_z)``. Exact for circular sections only.

        Returns:
            Shear stress in Pa, symbolic in ``x``.
        """
        c = max(self.section.c_y, self.section.c_z)
        return self._torque_expr() * c / self.section.polar_moment

    def bending_stress(self, dir="both"):
        """
        Outer-fiber bending stress sigma = M(x) * c / I per plane.

        Args:
            dir (str): "y", "z", or "both" (default) for a dict of both
                planes.

        Returns:
            Bending stress in Pa, symbolic in ``x`` — a single expression
            for dir "y"/"z", ``{"y": expr, "z": expr}`` for "both".

        Raises:
            ValueError: If dir is not "y", "z" or "both".
        """
        if dir not in ("y", "z", "both"):
            raise ValueError("dir must be 'y', 'z', or 'both'")
        moments = self.bending_moment()
        stresses = {
            "y": moments["y"] * self.section.c_y / self.section.second_moment_z,
            "z": moments["z"] * self.section.c_z / self.section.second_moment_y,
        }
        return stresses if dir == "both" else stresses[dir]

    # ---- Visualization ----

    def _sample(self, expr, n):
        """Sample a symbolic expression at n evenly spaced points on
        [0, length], returning (xs, ys) lists of floats."""
        if n < 2:
            raise ValueError("n must be at least 2")
        expr = sympify(expr)
        free = expr.free_symbols - {_X}
        if free:
            raise ValueError(
                f"result contains free symbols {sorted(map(str, free))}; "
                "all beam inputs must be numeric"
            )
        xs = [self.length * i / (n - 1) for i in range(n)]
        ys = [float(expr.subs(_X, x)) for x in xs]
        return xs, ys

    def _plot_curves(self, curves, ylabel, n, show, ax, feature):
        """Draw (expr, label, color) curves in the shared house style and
        return the figure."""
        plt = _pyplot(feature)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.figure
        for expr, label, color in curves:
            xs, ys = self._sample(expr, n)
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label)
        ax.axhline(0, color=_COLOR_GREY, linewidth=0.8, linestyle="--")
        ax.set_xlabel("position x [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(self.name or "Beam3D")
        ax.grid(True, color=_COLOR_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    @staticmethod
    def _plane_curves(results, dir):
        """(expr, label, color) rows for one plane or both."""
        if dir not in ("y", "z", "both"):
            raise ValueError("dir must be 'y', 'z', or 'both'")
        rows = {"y": (results["y"], "y", _COLOR_Y), "z": (results["z"], "z", _COLOR_Z)}
        return list(rows.values()) if dir == "both" else [rows[dir]]

    def plot_shear_force(self, dir="both", n=200, show=True, ax=None):
        """
        Plot the shear force diagram(s).

        Args:
            dir (str): "y", "z", or "both" (default).
            n (int): Number of sample points (>= 2).
            show (bool): Call ``plt.show()`` (default: True).
            ax (matplotlib.axes.Axes): Existing axes to draw into.

        Returns:
            matplotlib.figure.Figure: The figure.

        Raises:
            ImportError: If matplotlib is not installed.
            ValueError: If dir is invalid or n < 2.
        """
        curves = self._plane_curves(self.shear_force(), dir)
        return self._plot_curves(curves, "shear force [N]", n, show, ax, "plot_shear_force")

    def plot_bending_moment(self, dir="both", n=200, show=True, ax=None):
        """
        Plot the bending moment diagram(s). Same arguments and returns as
        :meth:`plot_shear_force`.
        """
        curves = self._plane_curves(self.bending_moment(), dir)
        return self._plot_curves(
            curves, "bending moment [N*m]", n, show, ax, "plot_bending_moment"
        )

    def plot_deflection(self, dir="both", n=200, show=True, ax=None):
        """
        Plot the deflection diagram(s). Same arguments and returns as
        :meth:`plot_shear_force`.
        """
        curves = self._plane_curves(self.deflection(), dir)
        return self._plot_curves(curves, "deflection [m]", n, show, ax, "plot_deflection")

    def plot_bending_stress(self, dir="both", n=200, show=True, ax=None):
        """
        Plot the outer-fiber bending stress diagram(s). Same arguments
        and returns as :meth:`plot_shear_force`.
        """
        curves = self._plane_curves(self.bending_stress("both"), dir)
        return self._plot_curves(
            curves, "bending stress [Pa]", n, show, ax, "plot_bending_stress"
        )

    def plot_axial_stress(self, n=200, show=True, ax=None):
        """
        Plot the axial stress diagram. Same arguments and returns as
        :meth:`plot_shear_force` (no ``dir``).
        """
        curves = [(self.axial_stress(), "axial", _COLOR_Y)]
        return self._plot_curves(curves, "axial stress [Pa]", n, show, ax, "plot_axial_stress")

    def plot_torsional_stress(self, n=200, show=True, ax=None):
        """
        Plot the torsional shear stress diagram. Same arguments and
        returns as :meth:`plot_shear_force` (no ``dir``).
        """
        curves = [(self.torsional_stress(), "torsion", _COLOR_Y)]
        return self._plot_curves(
            curves, "torsional stress [Pa]", n, show, ax, "plot_torsional_stress"
        )

    def plot_results(self, n=200, show=True, ax=None):
        """
        Combined results figure: stacked shear force, bending moment,
        axial force, torque and deflection panels sharing the x axis.

        Args:
            n (int): Number of sample points per curve (>= 2).
            show (bool): Call ``plt.show()`` (default: True).
            ax (tuple): Existing 5-tuple of axes to draw into. If None, a
                new figure with five stacked subplots is created.

        Returns:
            matplotlib.figure.Figure: The figure containing all panels.

        Raises:
            ImportError: If matplotlib is not installed.
            ValueError: If n < 2.
        """
        plt = _pyplot("plot_results")
        if ax is None:
            fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        else:
            axes = ax
            fig = axes[0].figure

        shear = self.shear_force()
        moment = self.bending_moment()
        deflection = self.deflection()
        panels = [
            (self._plane_curves(shear, "both"), "shear force [N]"),
            (self._plane_curves(moment, "both"), "bending moment [N*m]"),
            ([(self.axial_force(), "axial", _COLOR_Y)], "axial force [N]"),
            ([(self.torsional_moment(), "torque", _COLOR_Y)], "torque [N*m]"),
            (self._plane_curves(deflection, "both"), "deflection [m]"),
        ]
        for panel_ax, (curves, ylabel) in zip(axes, panels):
            for expr, label, color in curves:
                xs, ys = self._sample(expr, n)
                panel_ax.plot(xs, ys, color=color, linewidth=1.5, label=label)
            panel_ax.axhline(0, color=_COLOR_GREY, linewidth=0.8, linestyle="--")
            panel_ax.set_ylabel(ylabel)
            panel_ax.grid(True, color=_COLOR_GRID, linewidth=0.8)
            panel_ax.set_axisbelow(True)
            panel_ax.legend(loc="best", fontsize=8)
        axes[0].set_title(self.name or "Beam3D")
        axes[-1].set_xlabel("position x [m]")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def draw(self, show=True, ax=None):
        """
        Draw a flat pictorial schematic of the beam: body, supports and
        load glyphs. z-direction loads are drawn as dashed in-plane
        arrows labeled "(z)" — a flat schematic, not an isometric view.
        Glyph sizes are fixed relative to the beam length; forces are not
        drawn to scale.

        Args:
            show (bool): Call ``plt.show()`` (default: True).
            ax (matplotlib.axes.Axes): Existing axes to draw into.

        Returns:
            matplotlib.figure.Figure: The figure.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        plt = _pyplot("draw")
        from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.figure

        length = self.length
        h = length / 15

        ax.add_patch(Rectangle(
            (0, -h / 2), length, h, facecolor=_COLOR_GRID, edgecolor=_COLOR_Z,
            linewidth=1.5, zorder=1,
        ))

        for location, kind in self._supports:
            if kind == "fixed":
                ax.plot([location, location], [-1.2 * h, 1.2 * h],
                        color=_COLOR_Z, linewidth=3, zorder=2)
                hatch_dir = -1 if location > length / 2 else 1
                for frac in (-0.9, -0.3, 0.3, 0.9):
                    y0 = frac * h
                    ax.plot([location, location - hatch_dir * 0.4 * h],
                            [y0, y0 - 0.4 * h], color=_COLOR_Z, linewidth=1, zorder=2)
            else:
                ax.add_patch(Polygon(
                    [(location, -h / 2), (location - 0.4 * h, -1.3 * h),
                     (location + 0.4 * h, -1.3 * h)],
                    closed=True, facecolor=_COLOR_Z, zorder=2,
                ))
                if kind == "roller":
                    for dx in (-0.2 * h, 0.2 * h):
                        ax.add_patch(plt.Circle(
                            (location + dx, -1.45 * h), 0.13 * h,
                            facecolor=_COLOR_Z, zorder=2,
                        ))

        def _force_arrow(x_pos, value, color, linestyle, label):
            if value <= 0:  # acts downward: arrow pointing down onto the top face
                tail, tip = (x_pos, h / 2 + 2 * h), (x_pos, h / 2)
            else:
                tail, tip = (x_pos, -h / 2 - 2 * h), (x_pos, -h / 2)
            ax.annotate(
                "", xy=tip, xytext=tail,
                arrowprops={"arrowstyle": "-|>", "color": color, "linestyle": linestyle,
                            "linewidth": 1.5},
            )
            ax.text(x_pos, tail[1] + (0.15 * h if value <= 0 else -0.45 * h), label,
                    fontsize=8, color=color, ha="center")

        for value, start, order, dir_, end in self._loads:
            if dir_ == "x":
                x_dir = 1 if value >= 0 else -1
                ax.annotate(
                    "", xy=(start + x_dir * 1.5 * h, 0), xytext=(start, 0),
                    arrowprops={"arrowstyle": "-|>", "color": _COLOR_Y, "linewidth": 1.5},
                )
                ax.text(start, 0.25 * h, f"{value:g} N (x)", fontsize=8,
                        color=_COLOR_Y, ha="center")
            elif order == -1:
                color = _COLOR_Y if dir_ == "y" else _COLOR_GREY
                style = "-" if dir_ == "y" else "--"
                suffix = "" if dir_ == "y" else " (z)"
                _force_arrow(start, value, color, style, f"{value:g} N{suffix}")
            else:  # distributed: arrow row with a connecting tail line
                stop = length if end is None else end
                color = _COLOR_Y if dir_ == "y" else _COLOR_GREY
                style = "-" if dir_ == "y" else "--"
                suffix = "" if dir_ == "y" else " (z)"
                top = h / 2 + 1.2 * h if value <= 0 else -h / 2 - 1.2 * h
                face = h / 2 if value <= 0 else -h / 2
                arrow_n = 7
                for i in range(arrow_n):
                    x_pos = start + (stop - start) * i / (arrow_n - 1)
                    ax.annotate(
                        "", xy=(x_pos, face), xytext=(x_pos, top),
                        arrowprops={"arrowstyle": "-|>", "color": color,
                                    "linestyle": style, "linewidth": 1},
                    )
                ax.plot([start, stop], [top, top], color=color, linestyle=style, linewidth=1)
                ax.text((start + stop) / 2, top + (0.15 * h if value <= 0 else -0.45 * h),
                        f"{value:g} N/m{suffix}", fontsize=8, color=color, ha="center")

        for value, location, dir_ in self._moments:
            ax.add_patch(FancyArrowPatch(
                (location + 0.7 * h, -0.6 * h), (location + 0.7 * h, 0.6 * h),
                connectionstyle="arc3,rad=0.8", arrowstyle="-|>",
                mutation_scale=12, color=_COLOR_ALERT, zorder=3,
            ))
            ax.text(location + 1.3 * h, 0.8 * h, f"M={value:g} N*m ({dir_})",
                    fontsize=8, color=_COLOR_ALERT, ha="left")

        for value, location in self._torques:
            ax.add_patch(FancyArrowPatch(
                (location, 0.6 * h), (location, -0.6 * h),
                connectionstyle="arc3,rad=1.0", arrowstyle="-|>",
                mutation_scale=12, color=_COLOR_ALERT, zorder=3,
            ))
            ax.text(location, -1.1 * h, f"T={value:g} N*m",
                    fontsize=8, color=_COLOR_ALERT, ha="center", va="top")

        ax.set_aspect("equal", adjustable="box")
        ax.set_yticks([])
        ax.set_xlim(-0.06 * length, 1.14 * length)
        ax.set_ylim(-4 * h, 4 * h)
        ax.set_xlabel("position x [m]")
        ax.set_title(self.name or "Beam3D")
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def __repr__(self):
        return (
            f"Beam3D(length={self.length}m, section={self.section!r}, "
            f"material={self.material!r})"
        )
