"""Shaft keys (parallel / square keys).

Units convention: geometry in mm, torque in N*mm, forces in N and
stresses in MPa (N/mm^2), consistent with the other element modules.

Reference: standard machine-design key statics (Shigley Ch. 7). A key
transmits torque between a shaft and a hub. The tangential force at the
shaft surface ``F = 2T/d`` shears the key across its width and bears on
its flank; both modes are checked against yielding (shear against the
distortion-energy shear yield ``0.577*Sy``, bearing against ``Sy``).
"""

from ..base import MechaElement
from ..materials import get_material_properties
from ..utils.constants import SAFETY_FACTOR_STATIC

SHEAR_YIELD_FACTOR = 0.577  # distortion-energy shear yield, Ssy = 0.577*Sy


class Key(MechaElement):
    """
    Parallel (or square) shaft key transmitting torque.

    The key sits in a keyway on a shaft of diameter ``shaft_diameter``;
    the transmitted torque produces a tangential force at the shaft
    surface that the key resists in shear (across ``width * length``) and
    in bearing (on the flank, half the height in contact).

    Attributes:
        width (float): Key width w in mm. Settable.
        height (float): Key height h in mm. Settable.
        length (float): Key length L in mm. Settable.
        shaft_diameter (float): Shaft diameter d in mm. Settable.
        material (str): Key material type.
    """

    def __init__(self, width, height, length, shaft_diameter,
                 material="steel", name=None):
        """
        Initialize a Key object.

        Args:
            width (float): Key width w in mm.
            height (float): Key height h in mm.
            length (float): Key length L in mm.
            shaft_diameter (float): Shaft diameter d in mm.
            material (str): Key material type (default: "steel").
            name (str): Optional identifier for the key.

        Raises:
            ValueError: If any dimension is not strictly positive.
        """
        super().__init__(name=name, material=material)
        self.width = width
        self.height = height
        self.length = length
        self.shaft_diameter = shaft_diameter

    # ---- Settable primary inputs ----

    @property
    def width(self):
        """float: Key width w in mm."""
        return self._width

    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError("Key width must be strictly positive")
        self._width = value

    @property
    def height(self):
        """float: Key height h in mm."""
        return self._height

    @height.setter
    def height(self, value):
        if value <= 0:
            raise ValueError("Key height must be strictly positive")
        self._height = value

    @property
    def length(self):
        """float: Key length L in mm."""
        return self._length

    @length.setter
    def length(self, value):
        if value <= 0:
            raise ValueError("Key length must be strictly positive")
        self._length = value

    @property
    def shaft_diameter(self):
        """float: Shaft diameter d in mm."""
        return self._shaft_diameter

    @shaft_diameter.setter
    def shaft_diameter(self, value):
        if value <= 0:
            raise ValueError("Shaft diameter must be strictly positive")
        self._shaft_diameter = value

    # ---- Areas ----

    @property
    def shear_area(self):
        """float: Shear area w * L in mm^2."""
        return self.width * self.length

    @property
    def bearing_area(self):
        """float: Bearing area (h/2) * L in mm^2 (half the height bears)."""
        return (self.height / 2) * self.length

    # ---- Loads and stresses ----

    def tangential_force(self, torque):
        """
        Tangential force at the shaft surface, ``F = 2T / d``.

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Tangential force in N.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        self._check_torque(torque)
        return 2 * torque / self.shaft_diameter

    def shear_stress(self, torque):
        """
        Shear stress across the key, ``tau = F / (w * L)``.

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Shear stress in MPa.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        return self.tangential_force(torque) / self.shear_area

    def bearing_stress(self, torque):
        """
        Bearing (crushing) stress on the key flank, ``sigma = F / ((h/2) L)``.

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Bearing stress in MPa.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        return self.tangential_force(torque) / self.bearing_area

    # ---- Safety factors ----

    def shear_safety_factor(self, torque):
        """
        Safety factor of the key against shear yielding (0.577*Sy).

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Safety factor.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        sy = self.material_properties["yield_strength"] / 1e6
        return SHEAR_YIELD_FACTOR * sy / self.shear_stress(torque)

    def bearing_safety_factor(self, torque):
        """
        Safety factor of the key against bearing yielding (Sy).

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Safety factor.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        sy = self.material_properties["yield_strength"] / 1e6
        return sy / self.bearing_stress(torque)

    def torque_capacity(self, safety_factor=SAFETY_FACTOR_STATIC):
        """
        Largest torque the key carries at a given safety factor.

        Inverts both failure modes (shear and bearing, each linear in the
        torque) and returns the smaller governing torque.

        Args:
            safety_factor (float): Required safety factor (default: the
                static factor from :mod:`mecapy.utils.constants`).

        Returns:
            float: Allowable torque in N*mm.

        Raises:
            ValueError: If ``safety_factor`` is not strictly positive.
        """
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        reference = 1.0e6  # any torque works: stresses are linear in T
        governing = min(self.shear_safety_factor(reference),
                        self.bearing_safety_factor(reference))
        return reference * governing / safety_factor

    @staticmethod
    def _check_torque(torque):
        """Validate a transmitted torque."""
        if torque <= 0:
            raise ValueError("Torque must be strictly positive")

    def __repr__(self):
        return (
            f"Key(width={self.width}, height={self.height}, "
            f"length={self.length}, shaft_diameter={self.shaft_diameter}, "
            f"material={self.material!r})"
        )
