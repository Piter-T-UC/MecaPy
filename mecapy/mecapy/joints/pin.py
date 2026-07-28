"""Shear pins (dowel / clevis / shear-safety pins).

Units convention: geometry in mm, torque in N*mm, forces in N and
stresses in MPa (N/mm^2), consistent with the other element modules.

Reference: standard pin-connection statics (Shigley Ch. 7). A pin
carries a transverse load in shear across one or more planes (single or
double shear). A transverse pin through a shaft can also transmit torque:
the tangential force at the shaft surface ``F = 2T/d`` is the shear load.
Shear is checked against the distortion-energy shear yield ``0.577*Sy``.
"""

import math

from ..base import MechaElement

SHEAR_YIELD_FACTOR = 0.577  # distortion-energy shear yield, Ssy = 0.577*Sy


class Pin(MechaElement):
    """
    Shear pin (single or double shear).

    Attributes:
        diameter (float): Pin diameter d in mm. Settable.
        shear_planes (int): Number of shear planes (1 single shear,
            2 double shear). Settable.
        material (str): Pin material type.
    """

    def __init__(self, diameter, shear_planes=1, material="steel", name=None):
        """
        Initialize a Pin object.

        Args:
            diameter (float): Pin diameter d in mm.
            shear_planes (int): Number of shear planes (>= 1; default 1,
                single shear). Use 2 for a clevis (double shear).
            material (str): Pin material type (default: "steel").
            name (str): Optional identifier for the pin.

        Raises:
            ValueError: If ``diameter`` is not strictly positive or
                ``shear_planes`` is not an integer >= 1.
        """
        super().__init__(name=name, material=material)
        self.diameter = diameter
        self.shear_planes = shear_planes

    # ---- Settable primary inputs ----

    @property
    def diameter(self):
        """float: Pin diameter d in mm."""
        return self._diameter

    @diameter.setter
    def diameter(self, value):
        if value <= 0:
            raise ValueError("Diameter must be strictly positive")
        self._diameter = value

    @property
    def shear_planes(self):
        """int: Number of shear planes (1 single, 2 double shear)."""
        return self._shear_planes

    @shear_planes.setter
    def shear_planes(self, value):
        if value != int(value) or value < 1:
            raise ValueError("shear_planes must be an integer >= 1")
        self._shear_planes = int(value)

    # ---- Geometry ----

    @property
    def shear_area(self):
        """float: Total shear area (planes * pi*d^2/4) in mm^2."""
        return self.shear_planes * math.pi * self.diameter ** 2 / 4

    # ---- Direct transverse load ----

    def shear_stress(self, force):
        """
        Shear stress from a transverse load, ``tau = F / A_shear``.

        Args:
            force (float): Transverse load F in N.

        Returns:
            float: Shear stress in MPa.
        """
        return force / self.shear_area

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

    def allowable_force(self, safety_factor):
        """
        Largest transverse load the pin carries at a given safety factor.

        Args:
            safety_factor (float): Required safety factor.

        Returns:
            float: Allowable transverse force in N.

        Raises:
            ValueError: If ``safety_factor`` is not strictly positive.
        """
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        sy = self.material_properties["yield_strength"] / 1e6
        return SHEAR_YIELD_FACTOR * sy * self.shear_area / safety_factor

    # ---- Torque transmission (cross pin through a shaft) ----

    def torque_shear_stress(self, torque, shaft_diameter):
        """
        Shear stress when the pin transmits torque through a shaft.

        A transverse (cross) pin through a shaft of diameter ``d_shaft``
        reacts the tangential force ``F = 2T / d_shaft`` in shear.

        Args:
            torque (float): Transmitted torque in N*mm.
            shaft_diameter (float): Shaft diameter in mm.

        Returns:
            float: Shear stress in MPa.

        Raises:
            ValueError: If ``torque`` or ``shaft_diameter`` is not
                strictly positive.
        """
        if torque <= 0:
            raise ValueError("Torque must be strictly positive")
        if shaft_diameter <= 0:
            raise ValueError("Shaft diameter must be strictly positive")
        force = 2 * torque / shaft_diameter
        return self.shear_stress(force)

    def torque_safety_factor(self, torque, shaft_diameter):
        """
        Safety factor of a torque-transmitting cross pin (0.577*Sy).

        Args:
            torque (float): Transmitted torque in N*mm.
            shaft_diameter (float): Shaft diameter in mm.

        Returns:
            float: Safety factor.

        Raises:
            ValueError: If ``torque`` or ``shaft_diameter`` is not
                strictly positive.
        """
        sy = self.material_properties["yield_strength"] / 1e6
        return (SHEAR_YIELD_FACTOR * sy
                / self.torque_shear_stress(torque, shaft_diameter))

    def __repr__(self):
        return (
            f"Pin(diameter={self.diameter}, shear_planes={self.shear_planes}, "
            f"material={self.material!r})"
        )
