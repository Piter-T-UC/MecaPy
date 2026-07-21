"""Beam analysis built on top of SymPy's continuum-mechanics beam."""

from sympy import Rational, symbols
from sympy.physics.continuum_mechanics.beam import Beam as _SymBeam

from ..base import MechaElement


class Beam(MechaElement):
    """
    Beam analysis and design.

    This class wraps :class:`sympy.physics.continuum_mechanics.beam.Beam`
    to provide symbolic analysis of shear force, bending moment, slope and
    deflection, while inheriting the shared stress/safety-factor behaviour
    from :class:`~mecapy.base.MechaElement`.

    Supports and loads are added incrementally; reaction loads are solved
    automatically the first time a result is requested.

    Attributes:
        length (float): Beam length.
        elastic_modulus: Young's modulus (E). Defaults to the material's
            value from the database.
        second_moment: Second moment of area (I). Defaults to a symbolic
            ``I`` so results can be kept symbolic.
    """

    def __init__(self, length, material="steel", elastic_modulus=None,
                 second_moment=None, name=None):
        """
        Initialize a Beam.

        Args:
            length (float): Beam length (consistent length units).
            material (str): Material name (default: "steel").
            elastic_modulus: Young's modulus E in Pa. If ``None``, the
                material database value is used.
            second_moment: Second moment of area I. If ``None``, a symbolic
                ``I`` is used so results stay symbolic.
            name (str): Optional identifier for the beam.
        """
        super().__init__(name=name, material=material)
        self.length = length
        self.elastic_modulus = (
            elastic_modulus
            if elastic_modulus is not None
            else self.material_properties["elastic_modulus"]
        )
        self.second_moment = (
            second_moment if second_moment is not None else symbols("I", positive=True)
        )
        self._beam = _SymBeam(length, self.elastic_modulus, self.second_moment)
        self._reactions = []
        self._solved = False

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------
    def add_support(self, location, kind="pin"):
        """
        Add a support at a location.

        Args:
            location (float): Position of the support along the beam.
            kind (str): Support type - "pin", "roller" or "fixed".

        Returns:
            Beam: ``self`` to allow method chaining.
        """
        self._beam.apply_support(location, kind)
        self._reactions.append(symbols(f"R_{location}"))
        if kind == "fixed":
            self._reactions.append(symbols(f"M_{location}"))
        self._solved = False
        return self

    def add_load(self, value, start, order, end=None):
        """
        Add a load using SymPy's singularity-function convention.

        Args:
            value (float): Magnitude of the load (negative acts downward).
            start (float): Application point along the beam.
            order (int): Singularity order (-2 moment, -1 point load,
                0 uniform, 1 ramp, ...).
            end (float): Optional end position for distributed loads.

        Returns:
            Beam: ``self`` to allow method chaining.
        """
        self._beam.apply_load(value, start, order, end=end)
        self._solved = False
        return self

    def add_point_load(self, value, location):
        """Add a concentrated point load (order -1)."""
        return self.add_load(value, location, -1)

    def add_moment(self, value, location):
        """Add a concentrated moment (order -2)."""
        return self.add_load(value, location, -2)

    def add_distributed_load(self, value, start, end):
        """Add a uniformly distributed load (order 0) between two points."""
        return self.add_load(value, start, 0, end=end)
    
    def add_triangular_load(self, value, start, end):
        """Add a uniformly triangular load (order 1) between two points."""
        return self.add_load(value, start, 1, end=end)

    def add_function_load(self, func, start=0, end=None, segments=50):
        """
        Add a distributed load given as an arbitrary function q = f(x)
        (force per unit length), discretized into ``segments``
        piecewise-constant distributed loads — each segment carries the
        value of ``f`` at its midpoint. The approximation improves with
        more segments; 50 is ample for smooth load shapes.

        Segment positions are exact ``sympy.Rational`` values, since
        sympy's reaction solver breaks on plain-float positions.

        Note: with a symbolic ``E`` or ``I``, :meth:`max_bending_moment`
        / :meth:`max_deflection` fall back to sympy's symbolic extremum
        search, which is intractably slow with the many singularity
        terms this method adds — give the beam numeric ``E`` and ``I``
        so the fast sampled path applies.

        Args:
            func (callable): Load intensity ``f(x)`` — takes a position
                (float) and returns force per unit length (negative acts
                downward).
            start (float): Start of the loaded span (default: 0).
            end (float): End of the loaded span (default: the beam's
                full length).
            segments (int): Number of constant-load segments (>= 1,
                default: 50).

        Returns:
            Beam: ``self`` to allow method chaining.

        Raises:
            ValueError: If ``func`` is not callable, ``segments`` is not
                a positive integer, or the span is not within
                ``[0, length]``.
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
        start_r, end_r = Rational(start), Rational(end)
        width = (end_r - start_r) / segments
        for i in range(segments):
            x0 = start_r + i * width
            value = float(func(float(x0 + width / 2)))
            if value:
                self.add_load(value, x0, 0, end=x0 + width)
        return self

    def solve(self):
        """
        Solve for the unknown reaction loads.

        Returns:
            dict: Mapping of reaction symbols to their solved values.
        """
        self._beam.solve_for_reaction_loads(*self._reactions)
        self._solved = True
        return self._beam.reaction_loads

    def _ensure_solved(self):
        if not self._solved and self._reactions:
            self.solve()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    @property
    def reactions(self):
        """dict: Solved reaction loads."""
        self._ensure_solved()
        return self._beam.reaction_loads

    def shear_force(self):
        """Return the shear force expression along the beam."""
        self._ensure_solved()
        return self._beam.shear_force()

    def bending_moment(self):
        """Return the bending moment expression along the beam."""
        self._ensure_solved()
        return self._beam.bending_moment()

    def slope(self):
        """Return the slope expression along the beam."""
        self._ensure_solved()
        return self._beam.slope()

    def deflection(self):
        """Return the deflection expression along the beam."""
        self._ensure_solved()
        return self._beam.deflection()

    def _numeric_max(self, expr, n=201):
        """(location, magnitude) of the largest-magnitude sample of a
        fully numeric expression, or None if it has symbols besides x."""
        x = self._beam.variable
        if expr.free_symbols - {x}:
            return None
        xs = [self.length * i / (n - 1) for i in range(n)]
        return max(
            ((xi, abs(float(expr.subs(x, xi)))) for xi in xs),
            key=lambda point: point[1],
        )

    def max_bending_moment(self):
        """
        Return ``(location, value)`` of the maximum bending moment.

        When the moment expression is fully numeric the extremum is
        found by sampling 201 points and ``value`` is the magnitude —
        sympy's symbolic search takes minutes with more than a couple of
        loads. With symbolic inputs (e.g. symbolic E or I) sympy's exact
        ``max_bmoment`` is used instead.
        """
        self._ensure_solved()
        numeric = self._numeric_max(self._beam.bending_moment())
        return numeric if numeric is not None else self._beam.max_bmoment()

    def max_deflection(self):
        """
        Return ``(location, value)`` of the maximum deflection.

        Numeric-vs-symbolic behaviour matches
        :meth:`max_bending_moment`.
        """
        self._ensure_solved()
        numeric = self._numeric_max(self._beam.deflection())
        return numeric if numeric is not None else self._beam.max_deflection()

    def bending_stress(self, distance_to_fiber, second_moment=None):
        """
        Calculate the maximum bending stress using ``sigma = M * c / I``.

        Args:
            distance_to_fiber (float): Distance ``c`` from the neutral axis
                to the outermost fiber.
            second_moment: Second moment of area ``I``. Defaults to the
                beam's ``second_moment``.

        Returns:
            The bending stress (symbolic unless ``E``, ``I`` and the
            geometry are all numeric).
        """
        inertia = second_moment if second_moment is not None else self.second_moment
        _, moment = self.max_bending_moment()
        return moment * distance_to_fiber / inertia

    def __repr__(self):
        return f"Beam(length={self.length}, material={self.material!r})"
