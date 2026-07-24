"""Riveted (and equivalent shear) fasteners.

Units convention: geometry in mm, forces in N and stresses in MPa
(N/mm^2), consistent with the other element modules.

Reference: standard riveted-joint statics (Shigley Ch. 8 shear
connections). A rivet is, mechanically, a bolt loaded in shear without
preload: it carries a transverse load in shear across its shank and in
bearing against the connected plate. Shear is checked against the
distortion-energy shear yield ``0.577*Sy`` and bearing against ``Sy``.
"""

import math

from ..base import MechaElement
from ..utils.constants import SAFETY_FACTOR_STATIC

SHEAR_YIELD_FACTOR = 0.577  # distortion-energy shear yield, Ssy = 0.577*Sy


class Rivet(MechaElement):
    """
    Rivet in a shear connection (a preload-free bolt in shear/bearing).

    Attributes:
        diameter (float): Rivet shank diameter d in mm. Settable.
        plate_thickness (float): Bearing plate thickness t in mm.
            Settable.
        shear_planes (int): Number of shear planes (1 single shear,
            2 double shear). Settable.
        material (str): Rivet material type.
    """

    def __init__(self, diameter, plate_thickness, shear_planes=1,
                 material="steel", name=None):
        """
        Initialize a Rivet object.

        Args:
            diameter (float): Rivet shank diameter d in mm.
            plate_thickness (float): Thickness t of the plate the rivet
                bears against, in mm.
            shear_planes (int): Number of shear planes (>= 1; default 1).
            material (str): Rivet material type (default: "steel").
            name (str): Optional identifier for the rivet.

        Raises:
            ValueError: If a dimension is not strictly positive or
                ``shear_planes`` is not an integer >= 1.
        """
        super().__init__(name=name, material=material)
        self.diameter = diameter
        self.plate_thickness = plate_thickness
        self.shear_planes = shear_planes

    # ---- Settable primary inputs ----

    @property
    def diameter(self):
        """float: Rivet shank diameter d in mm."""
        return self._diameter

    @diameter.setter
    def diameter(self, value):
        if value <= 0:
            raise ValueError("Diameter must be strictly positive")
        self._diameter = value

    @property
    def plate_thickness(self):
        """float: Bearing plate thickness t in mm."""
        return self._plate_thickness

    @plate_thickness.setter
    def plate_thickness(self, value):
        if value <= 0:
            raise ValueError("Plate thickness must be strictly positive")
        self._plate_thickness = value

    @property
    def shear_planes(self):
        """int: Number of shear planes (1 single, 2 double shear)."""
        return self._shear_planes

    @shear_planes.setter
    def shear_planes(self, value):
        if value != int(value) or value < 1:
            raise ValueError("shear_planes must be an integer >= 1")
        self._shear_planes = int(value)

    # ---- Areas ----

    @property
    def shear_area(self):
        """float: Total shear area (planes * pi*d^2/4) in mm^2."""
        return self.shear_planes * math.pi * self.diameter ** 2 / 4

    @property
    def bearing_area(self):
        """float: Projected bearing area d * t in mm^2."""
        return self.diameter * self.plate_thickness

    # ---- Stresses ----

    def shear_stress(self, force):
        """
        Shear stress in the rivet shank, ``tau = F / A_shear``.

        Args:
            force (float): Transverse load F in N.

        Returns:
            float: Shear stress in MPa.
        """
        return force / self.shear_area

    def bearing_stress(self, force):
        """
        Bearing stress against the plate, ``sigma = F / (d * t)``.

        Args:
            force (float): Transverse load F in N.

        Returns:
            float: Bearing stress in MPa.
        """
        return force / self.bearing_area

    # ---- Safety factors ----

    def shear_safety_factor(self, force):
        """
        Safety factor against shear yielding (0.577*Sy).

        Args:
            force (float): Transverse load F in N.

        Returns:
            float: Safety factor.

        Raises:
            ValueError: If ``force`` is zero.
        """
        if force == 0:
            raise ValueError("Force must be non-zero to compute a safety factor")
        sy = self.material_properties["yield_strength"] / 1e6
        return SHEAR_YIELD_FACTOR * sy / self.shear_stress(abs(force))

    def bearing_safety_factor(self, force):
        """
        Safety factor against bearing yielding (Sy).

        Args:
            force (float): Transverse load F in N.

        Returns:
            float: Safety factor.

        Raises:
            ValueError: If ``force`` is zero.
        """
        if force == 0:
            raise ValueError("Force must be non-zero to compute a safety factor")
        sy = self.material_properties["yield_strength"] / 1e6
        return sy / self.bearing_stress(abs(force))

    def allowable_force(self, safety_factor=SAFETY_FACTOR_STATIC):
        """
        Largest transverse load the rivet carries at a given safety factor.

        Inverts both modes (shear and bearing, each linear in the load)
        and returns the smaller governing force.

        Args:
            safety_factor (float): Required safety factor (default: the
                static factor from :mod:`mecapy.utils.constants`).

        Returns:
            float: Allowable transverse force in N.

        Raises:
            ValueError: If ``safety_factor`` is not strictly positive.
        """
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        reference = 1.0e4  # any load works: stresses are linear in F
        governing = min(self.shear_safety_factor(reference),
                        self.bearing_safety_factor(reference))
        return reference * governing / safety_factor

    def __repr__(self):
        return (
            f"Rivet(diameter={self.diameter}, "
            f"plate_thickness={self.plate_thickness}, "
            f"shear_planes={self.shear_planes}, material={self.material!r})"
        )
