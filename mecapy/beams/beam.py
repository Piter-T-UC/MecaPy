"""Beam analysis built on top of SymPy's continuum-mechanics beam."""

from sympy import symbols
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

    LATEX_FIELDS = [
        ("length", "Length", "m"),
        ("elastic_modulus", "Elastic modulus $E$", "Pa"),
        ("second_moment", "Second moment $I$", "m$^4$"),
    ]

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

    def max_bending_moment(self):
        """Return ``(location, value)`` of the maximum bending moment."""
        self._ensure_solved()
        return self._beam.max_bmoment()

    def max_deflection(self):
        """Return ``(location, value)`` of the maximum deflection."""
        self._ensure_solved()
        return self._beam.max_deflection()

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
