"""Gear design and analysis module.

Defines the :class:`Gear` base class shared by all gear types
(spur, helical, herringbone, bevel, worm wheel, ...). Geometry follows
the standard full-depth tooth system.

Units: SI-metric is primary (module in mm, angles in degrees). US
customary users can pass ``diametral_pitch`` (teeth/inch) instead of
``module``; it is converted internally so all downstream code is metric.
See :mod:`mecapy.utils.converters` for force/stress/power conversions.
"""

import math

from ..base import MechaElement


class Gear(MechaElement):
    """
    Base class for gears (standard full-depth involute teeth).

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`. Direct use is equivalent to a
    generic cylindrical gear; prefer the specific subclasses
    (:class:`~mecapy.gears.SpurGear`, :class:`~mecapy.gears.HelicalGear`,
    ...) for real work.

    Attributes:
        teeth (int): Number of teeth.
        module (float): Gear module in mm.
        pressure_angle (float): Pressure angle in degrees (default 20).
        material (str): Material type.
    """

    #: Minimum number of teeth accepted by the constructor. Subclasses
    #: override (e.g. cylindrical gears require more; worms allow 1 start).
    _min_teeth = 1

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 material="steel", name=None, diametral_pitch=None):
        """
        Initialize a Gear object.

        Args:
            teeth (int): Number of teeth. Must be >= 1.
            module (float): Gear module in mm. Give exactly one of
                ``module`` or ``diametral_pitch``.
            pressure_angle (float): Pressure angle in degrees
                (default: 20). Must satisfy 0 < angle < 45.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the gear.
            diametral_pitch (float): US-customary alternative to
                ``module``, in teeth per inch. Stored internally as
                ``module = 25.4 / diametral_pitch``.

        Raises:
            ValueError: If teeth/module/pressure angle are non-physical,
                or if both (or neither) of ``module`` and
                ``diametral_pitch`` are given.
        """
        super().__init__(name=name, material=material)
        if teeth != int(teeth) or teeth < self._min_teeth:
            raise ValueError(
                f"Teeth must be an integer >= {self._min_teeth}, got {teeth}"
            )
        if (module is None) == (diametral_pitch is None):
            raise ValueError(
                "Give exactly one of 'module' (mm) or 'diametral_pitch' (1/in)"
            )
        if diametral_pitch is not None:
            if diametral_pitch <= 0:
                raise ValueError("Diametral pitch must be strictly positive")
            module = 25.4 / diametral_pitch
        if module <= 0:
            raise ValueError("Module must be strictly positive")
        if not 0 < pressure_angle < 45:
            raise ValueError("Pressure angle must be between 0 and 45 degrees")
        self.teeth = int(teeth)
        self.module = module
        self.pressure_angle = pressure_angle

    # ------------------------------------------------------------------
    # Geometry (standard full-depth system, dimensions in mm)
    # ------------------------------------------------------------------

    @property
    def diametral_pitch(self):
        """float: Diametral pitch in teeth per inch (25.4 / module)."""
        return 25.4 / self.module

    @property
    def pitch_diameter(self):
        """float: Pitch diameter in mm (teeth * module)."""
        return self.teeth * self.module

    @property
    def circular_pitch(self):
        """float: Circular pitch in mm (pi * module)."""
        return math.pi * self.module

    @property
    def addendum(self):
        """float: Addendum in mm (equal to the module)."""
        return self.module

    @property
    def dedendum(self):
        """float: Dedendum in mm (1.25 * module)."""
        return 1.25 * self.module

    @property
    def outside_diameter(self):
        """float: Outside (tip) diameter in mm."""
        return self.pitch_diameter + 2 * self.addendum

    @property
    def root_diameter(self):
        """float: Root diameter in mm."""
        return self.pitch_diameter - 2 * self.dedendum

    @property
    def whole_depth(self):
        """float: Whole tooth depth in mm (2.25 * module)."""
        return 2.25 * self.module

    @property
    def base_diameter(self):
        """float: Base circle diameter in mm (d * cos(pressure angle))."""
        return self.pitch_diameter * math.cos(math.radians(self.pressure_angle))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(teeth={self.teeth}, "
            f"module={self.module}, material={self.material!r})"
        )
