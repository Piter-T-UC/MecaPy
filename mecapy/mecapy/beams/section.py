"""Cross-section geometry for 3D beam analysis.

Units convention: consistent SI — lengths in m, areas in m^2, second
moments of area in m^4 — matching the beams module (Pa, m).
"""

import math


class CrossSection:
    """
    Cross-section properties consumed by :class:`~mecapy.beams.Beam3D`.

    Pure geometry data — it carries no material and no stress behaviour,
    so it deliberately does not inherit :class:`~mecapy.base.MechaElement`.
    All six inputs are validated to be strictly positive numbers; every
    value is stored as a plain float so beam results downstream stay
    numeric in the position variable ``x``.

    Coordinate convention (matching :class:`~mecapy.beams.Beam3D`): the
    beam axis is x; loads in y bend about z (use ``second_moment_z``),
    loads in z bend about y (use ``second_moment_y``).

    Attributes:
        area (float): Cross-sectional area in m^2. Settable.
        second_moment_y (float): Second moment of area about y, I_y, in
            m^4 (resists bending from z-direction loads). Settable.
        second_moment_z (float): Second moment of area about z, I_z, in
            m^4 (resists bending from y-direction loads). Settable.
        polar_moment (float): Torsion constant used as J in m^4. Derived —
            defaults to the polar second moment ``I_y + I_z``
            (perpendicular-axis theorem) and recomputes from I_y/I_z on
            every access. For a non-circular section the true St-Venant
            torsion constant J differs from the polar moment; assign a value
            to override the default with the real J (e.g. Roark's J for a
            rectangle, as :meth:`rectangular` does). Assign ``None`` to drop
            the override and return to the derived polar moment.
        c_y (float): Outer-fiber distance from the neutral axis in the
            y direction, in m. Settable.
        c_z (float): Outer-fiber distance from the neutral axis in the
            z direction, in m. Settable.
    """

    def __init__(self, area, second_moment_y, second_moment_z, c_y, c_z):
        """
        Initialize a CrossSection.

        Args:
            area (float): Cross-sectional area in m^2. Must be strictly
                positive.
            second_moment_y (float): I_y in m^4. Must be strictly positive.
            second_moment_z (float): I_z in m^4. Must be strictly positive.
            c_y (float): Outer-fiber distance in y, in m. Must be strictly
                positive.
            c_z (float): Outer-fiber distance in z, in m. Must be strictly
                positive.

        Raises:
            ValueError: If any input is not a strictly positive number.
        """
        self._polar_override = None
        self.area = area
        self.second_moment_y = second_moment_y
        self.second_moment_z = second_moment_z
        self.c_y = c_y
        self.c_z = c_z

    @staticmethod
    def _positive_float(value, label):
        """Validate and return ``value`` as a strictly positive float."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number, got {value!r}")
        if value <= 0:
            raise ValueError(f"{label} must be strictly positive")
        return value

    @property
    def area(self):
        """float: Cross-sectional area in m^2."""
        return self._area

    @area.setter
    def area(self, value):
        self._area = self._positive_float(value, "area")

    @property
    def second_moment_y(self):
        """float: Second moment of area about y, I_y, in m^4."""
        return self._second_moment_y

    @second_moment_y.setter
    def second_moment_y(self, value):
        self._second_moment_y = self._positive_float(value, "second_moment_y")

    @property
    def second_moment_z(self):
        """float: Second moment of area about z, I_z, in m^4."""
        return self._second_moment_z

    @second_moment_z.setter
    def second_moment_z(self, value):
        self._second_moment_z = self._positive_float(value, "second_moment_z")

    @property
    def polar_moment(self):
        """float: Torsion constant J in m^4.

        The explicit override if one was set, else the derived polar second
        moment ``I_y + I_z`` (recomputed from the current I_y/I_z).
        """
        if self._polar_override is not None:
            return self._polar_override
        return self.second_moment_y + self.second_moment_z

    @polar_moment.setter
    def polar_moment(self, value):
        # None clears the override, reverting to the derived I_y + I_z.
        if value is None:
            self._polar_override = None
        else:
            self._polar_override = self._positive_float(value, "polar_moment")

    @property
    def c_y(self):
        """float: Outer-fiber distance in the y direction, in m."""
        return self._c_y

    @c_y.setter
    def c_y(self, value):
        self._c_y = self._positive_float(value, "c_y")

    @property
    def c_z(self):
        """float: Outer-fiber distance in the z direction, in m."""
        return self._c_z

    @c_z.setter
    def c_z(self, value):
        self._c_z = self._positive_float(value, "c_z")

    @classmethod
    def circular(cls, diameter):
        """
        Build the section of a solid circular bar.

        ``A = pi*d^2/4``, ``I_y = I_z = pi*d^4/64``, ``J = pi*d^4/32``,
        ``c_y = c_z = d/2``. The only section for which the torsional
        stress formula ``tau = T*c/J`` used by ``Beam3D`` is exact.

        Args:
            diameter (float): Diameter in m. Must be strictly positive.

        Returns:
            CrossSection: The circular section.

        Raises:
            ValueError: If ``diameter`` is not strictly positive.
        """
        d = cls._positive_float(diameter, "diameter")
        second_moment = math.pi * d ** 4 / 64
        return cls(
            area=math.pi * d ** 2 / 4,
            second_moment_y=second_moment,
            second_moment_z=second_moment,
            c_y=d / 2,
            c_z=d / 2,
        )

    @classmethod
    def rectangular(cls, width, height):
        """
        Build the section of a solid rectangular bar.

        ``width`` is the z-extent (b), ``height`` the y-extent (h):
        ``A = b*h``, ``I_z = b*h^3/12``, ``I_y = h*b^3/12``, ``c_y = h/2``,
        ``c_z = b/2``. J uses Roark's approximation for a solid rectangle,
        ``J = a*b^3*(16/3 - 3.36*(b/a)*(1 - b^4/(12*a^4)))/16`` with
        ``a >= b`` the half side lengths. Note that ``Beam3D``'s torsional
        stress ``tau = T*c/J`` is exact only for circular sections and is
        an approximation here.

        Args:
            width (float): Section width (z-extent) in m. Must be strictly
                positive.
            height (float): Section height (y-extent) in m. Must be
                strictly positive.

        Returns:
            CrossSection: The rectangular section.

        Raises:
            ValueError: If ``width`` or ``height`` is not strictly
                positive.
        """
        b = cls._positive_float(width, "width")
        h = cls._positive_float(height, "height")
        section = cls(
            area=b * h,
            second_moment_y=h * b ** 3 / 12,
            second_moment_z=b * h ** 3 / 12,
            c_y=h / 2,
            c_z=b / 2,
        )
        half_long, half_short = max(b, h) / 2, min(b, h) / 2
        # Roark's torsion constant replaces the I_y + I_z default, which
        # overestimates torsional stiffness for non-circular sections.
        section.polar_moment = (
            half_long * half_short ** 3
            * (16 / 3 - 3.36 * (half_short / half_long)
               * (1 - half_short ** 4 / (12 * half_long ** 4)))
        )
        return section

    def __repr__(self):
        return (
            f"CrossSection(area={self.area:.4g}, I_y={self.second_moment_y:.4g}, "
            f"I_z={self.second_moment_z:.4g}, J={self.polar_moment:.4g})"
        )
