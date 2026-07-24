"""Column (strut) buckling analysis.

Units convention: geometry in mm, area in mm^2, second moment of area in
mm^4, forces in N and stresses in MPa (N/mm^2), consistent with the
other element modules (shafts, bolts, gears).

Reference: Shigley's *Mechanical Engineering Design*, Ch. 4 (columns).
A column carries compression and may fail by elastic (Euler) buckling
when slender or by inelastic yielding described by the parabolic J.B.
Johnson formula when intermediate. The two regimes meet at the tangent
(transition) slenderness ``(Le/k)_1 = sqrt(2 pi^2 E / Sy)``. Eccentric
loading is handled by the secant formula.

This is the reusable column element; :class:`~mecapy.shafts.PowerScrew`
composes it for its own buckling check on the screw root section.
"""

import math

from ..base import MechaElement


class Column(MechaElement):
    """
    Compression column / strut with Euler and Johnson buckling.

    The cross-section is described by its area and its second moment of
    area about the buckling axis (so the class is section-agnostic —
    use :meth:`circular` or :meth:`rectangular` for the common shapes).
    The effective-length factor ``end_condition`` (K) captures the end
    restraints: 1.0 pinned-pinned, 0.5 fixed-fixed, 0.707 fixed-pinned,
    2.0 fixed-free.

    Attributes:
        length (float): Column length L in mm. Settable.
        area (float): Cross-sectional area A in mm^2. Settable.
        second_moment (float): Second moment of area I about the
            buckling axis in mm^4. Settable.
        end_condition (float): Effective-length factor K. Settable.
        material (str): Material type.
    """

    def __init__(self, length, area, second_moment, end_condition=1.0,
                 material="steel", name=None):
        """
        Initialize a Column object.

        Args:
            length (float): Column length L in mm.
            area (float): Cross-sectional area A in mm^2.
            second_moment (float): Second moment of area I about the
                buckling axis in mm^4.
            end_condition (float): Effective-length factor K (default
                1.0, pinned-pinned).
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the column.

        Raises:
            ValueError: If any dimension or the end-condition factor is
                not strictly positive.
        """
        super().__init__(name=name, material=material)
        self.length = length
        self.area = area
        self.second_moment = second_moment
        self.end_condition = end_condition

    # ---- Alternate constructors ----

    @classmethod
    def circular(cls, diameter, length, end_condition=1.0, material="steel",
                 name=None):
        """
        Build a column with a solid circular cross-section.

        Args:
            diameter (float): Section diameter d in mm.
            length (float): Column length L in mm.
            end_condition (float): Effective-length factor K (default 1.0).
            material (str): Material type (default: "steel").
            name (str): Optional identifier.

        Returns:
            Column: A column with ``area = pi*d^2/4`` and
            ``second_moment = pi*d^4/64``.

        Raises:
            ValueError: If ``diameter`` is not strictly positive.
        """
        if diameter <= 0:
            raise ValueError("Diameter must be strictly positive")
        area = math.pi * diameter ** 2 / 4
        second_moment = math.pi * diameter ** 4 / 64
        return cls(length, area, second_moment, end_condition=end_condition,
                   material=material, name=name)

    @classmethod
    def rectangular(cls, width, depth, length, end_condition=1.0,
                    material="steel", name=None):
        """
        Build a column with a solid rectangular cross-section.

        The weaker (smaller-I) axis governs buckling, so the second
        moment is taken about the axis parallel to the larger side.

        Args:
            width (float): Section width b in mm.
            depth (float): Section depth h in mm.
            length (float): Column length L in mm.
            end_condition (float): Effective-length factor K (default 1.0).
            material (str): Material type (default: "steel").
            name (str): Optional identifier.

        Returns:
            Column: A column whose ``second_moment`` is the smaller of
            the two principal second moments (weak-axis buckling).

        Raises:
            ValueError: If ``width`` or ``depth`` is not strictly positive.
        """
        if width <= 0 or depth <= 0:
            raise ValueError("Width and depth must be strictly positive")
        area = width * depth
        i_weak = min(width * depth ** 3, depth * width ** 3) / 12
        return cls(length, area, i_weak, end_condition=end_condition,
                   material=material, name=name)

    # ---- Settable primary inputs ----

    @property
    def length(self):
        """float: Column length L in mm."""
        return self._length

    @length.setter
    def length(self, value):
        if value <= 0:
            raise ValueError("Length must be strictly positive")
        self._length = value

    @property
    def area(self):
        """float: Cross-sectional area A in mm^2."""
        return self._area

    @area.setter
    def area(self, value):
        if value <= 0:
            raise ValueError("Area must be strictly positive")
        self._area = value

    @property
    def second_moment(self):
        """float: Second moment of area I about the buckling axis in mm^4."""
        return self._second_moment

    @second_moment.setter
    def second_moment(self, value):
        if value <= 0:
            raise ValueError("Second moment of area must be strictly positive")
        self._second_moment = value

    @property
    def end_condition(self):
        """float: Effective-length factor K (1.0 pinned-pinned)."""
        return self._end_condition

    @end_condition.setter
    def end_condition(self, value):
        if value <= 0:
            raise ValueError("End-condition factor must be strictly positive")
        self._end_condition = value

    # ---- Material (Pa -> MPa) ----

    @property
    def elastic_modulus(self):
        """float: Elastic modulus of the material in MPa."""
        return self.material_properties["elastic_modulus"] / 1e6

    @property
    def yield_strength(self):
        """float: Yield strength of the material in MPa."""
        return self.material_properties["yield_strength"] / 1e6

    # ---- Slenderness ----

    @property
    def effective_length(self):
        """float: Effective length Le = K * L in mm."""
        return self.end_condition * self.length

    @property
    def radius_of_gyration(self):
        """float: Radius of gyration k = sqrt(I / A) in mm."""
        return math.sqrt(self.second_moment / self.area)

    @property
    def slenderness_ratio(self):
        """float: Slenderness ratio Le / k (dimensionless)."""
        return self.effective_length / self.radius_of_gyration

    @property
    def transition_slenderness(self):
        """float: Euler/Johnson tangent slenderness (Le/k)_1.

        ``sqrt(2 pi^2 E / Sy)``. At or above this the Euler formula
        governs; below it the parabolic Johnson formula governs.
        """
        return math.sqrt(2 * math.pi ** 2 * self.elastic_modulus
                         / self.yield_strength)

    @property
    def is_slender(self):
        """bool: True when Euler (elastic) buckling governs."""
        return self.slenderness_ratio >= self.transition_slenderness

    # ---- Critical loads ----

    @property
    def euler_load(self):
        """float: Euler critical buckling load in N.

        ``P_cr = pi^2 E I / Le^2``. Valid only for slender columns
        (``is_slender`` True); for shorter columns it over-predicts the
        capacity — use :attr:`critical_load`, which selects the regime.
        """
        return (math.pi ** 2 * self.elastic_modulus * self.second_moment
                / self.effective_length ** 2)

    @property
    def johnson_load(self):
        """float: J.B. Johnson critical load in N (intermediate columns).

        ``P_cr = A [Sy - (Sy (Le/k) / (2 pi))^2 / E]`` — the parabola
        tangent to the Euler curve at :attr:`transition_slenderness`.
        """
        sy = self.yield_strength
        term = (sy * self.slenderness_ratio / (2 * math.pi)) ** 2 / self.elastic_modulus
        return self.area * (sy - term)

    @property
    def critical_load(self):
        """float: Governing critical load in N (Euler or Johnson).

        Selects the Euler load for slender columns and the Johnson load
        for intermediate ones, using :attr:`transition_slenderness`.
        """
        return self.euler_load if self.is_slender else self.johnson_load

    @property
    def critical_stress(self):
        """float: Critical load divided by area, in MPa."""
        return self.critical_load / self.area

    # ---- Safety factors ----

    def buckling_safety_factor(self, load):
        """
        Safety factor against buckling for a compressive load.

        Uses the governing :attr:`critical_load` (Euler or Johnson). Named
        to avoid shadowing the inherited stress-based
        :meth:`~mecapy.base.MechaElement.safety_factor`.

        Args:
            load (float): Applied compressive load in N (magnitude).

        Returns:
            float: Ratio of the critical load to the applied load.

        Raises:
            ValueError: If ``load`` is not strictly positive.
        """
        if load <= 0:
            raise ValueError("Load must be strictly positive")
        return self.critical_load / load

    def euler_safety_factor(self, load):
        """
        Safety factor against Euler buckling specifically.

        Args:
            load (float): Applied compressive load in N (magnitude).

        Returns:
            float: Ratio of the Euler load to the applied load.

        Raises:
            ValueError: If ``load`` is not strictly positive.
        """
        if load <= 0:
            raise ValueError("Load must be strictly positive")
        return self.euler_load / load

    # ---- Eccentric loading ----

    def secant_max_stress(self, load, eccentricity, extreme_fiber_distance):
        """
        Maximum compressive stress from the secant column formula.

        For a load ``P`` applied at eccentricity ``e`` from the section
        centroid (Shigley Eq. 4-50)::

            sigma_max = (P/A) * [1 + (e*c/k^2) * sec( (Le/2k) sqrt(P/(A E)) )]

        with ``c`` the distance from the centroid to the extreme fibre on
        the concave side.

        Args:
            load (float): Applied compressive load P in N.
            eccentricity (float): Load eccentricity e in mm.
            extreme_fiber_distance (float): Distance c from the centroid
                to the extreme fibre in mm.

        Returns:
            float: Maximum compressive stress in MPa.

        Raises:
            ValueError: If ``load`` is not strictly positive, or ``e`` or
                ``c`` is negative.
        """
        if load <= 0:
            raise ValueError("Load must be strictly positive")
        if eccentricity < 0 or extreme_fiber_distance < 0:
            raise ValueError("Eccentricity and fibre distance must be non-negative")
        k2 = self.radius_of_gyration ** 2
        phi = (self.effective_length / (2 * self.radius_of_gyration)) * math.sqrt(
            load / (self.area * self.elastic_modulus)
        )
        return (load / self.area) * (
            1 + (eccentricity * extreme_fiber_distance / k2) / math.cos(phi)
        )

    def __repr__(self):
        return (
            f"Column(length={self.length}, area={self.area}, "
            f"second_moment={self.second_moment}, "
            f"end_condition={self.end_condition}, material={self.material!r})"
        )
