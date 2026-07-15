"""Worm drives (worm + worm wheel).

A worm drive transmits power between perpendicular non-intersecting
shafts with a high reduction ratio (wheel teeth / worm starts). The worm
is a screw; its size is set by the axial module and an independent pitch
diameter.

Scope: geometry, kinematics, efficiency, self-locking check and a rough
Buckingham-style wear load — educational checks, not an AGMA 6034 worm
rating.
"""

import math

from ..base import MechaElement
from .gear import Gear


class Worm(MechaElement):
    """
    Cylindrical worm (the screw member of a worm drive).

    Attributes:
        starts (int): Number of threads (starts).
        module (float): Axial module in mm (equals the wheel transverse
            module for a 90-degree drive).
        pitch_diameter (float): Worm pitch diameter in mm. Independent of
            teeth/module; a common proportion is
            ``C**0.875 / 3 <= d <= C**0.875 / 1.7`` with C the center
            distance (Shigley).
        pressure_angle (float): Normal pressure angle in degrees.
        material (str): Material type.
    """

    def __init__(self, starts, module=None, pitch_diameter=None,
                 pressure_angle=20.0, material="steel", name=None):
        """
        Initialize a Worm object.

        Args:
            starts (int): Number of threads (>= 1).
            module (float): Axial module in mm.
            pitch_diameter (float): Worm pitch diameter in mm. Required.
            pressure_angle (float): Normal pressure angle in degrees
                (default: 20).
            material (str): Material type (default: "steel").
            name (str): Optional identifier.

        Raises:
            ValueError: For non-physical inputs.
        """
        super().__init__(name=name, material=material)
        if starts != int(starts) or starts < 1:
            raise ValueError("Starts must be an integer >= 1")
        if module is None or module <= 0:
            raise ValueError("Module must be given and strictly positive")
        if pitch_diameter is None or pitch_diameter <= 0:
            raise ValueError(
                "Pitch diameter must be given and strictly positive"
            )
        if not 0 < pressure_angle < 45:
            raise ValueError("Pressure angle must be between 0 and 45 degrees")
        self.starts = int(starts)
        self.module = module
        self.pitch_diameter = pitch_diameter
        self.pressure_angle = pressure_angle

    @property
    def lead(self):
        """float: Lead in mm (starts * pi * module) — axial advance per
        worm revolution."""
        return self.starts * math.pi * self.module

    @property
    def lead_angle(self):
        """float: Lead angle in degrees, atan(lead / (pi * d))."""
        return math.degrees(math.atan(
            self.lead / (math.pi * self.pitch_diameter)
        ))

    # ------------------------------------------------------------------
    # Drive-level calculations (worm + wheel)
    # ------------------------------------------------------------------

    def ratio_with(self, wheel):
        """
        Speed reduction ratio of the drive.

        Args:
            wheel (WormWheel): Mating worm wheel.

        Returns:
            float: Ratio (wheel teeth / worm starts).

        Raises:
            ValueError: If worm and wheel are not compatible.
        """
        from .transmission import _check_mesh

        _check_mesh(self, wheel)
        return wheel.teeth / self.starts

    def center_distance_with(self, wheel):
        """
        Center distance of the drive.

        Args:
            wheel (WormWheel): Mating worm wheel.

        Returns:
            float: Center distance in mm ((d_worm + d_wheel) / 2).
        """
        from .transmission import _check_mesh

        _check_mesh(self, wheel)
        return (self.pitch_diameter + wheel.pitch_diameter) / 2

    def efficiency(self, friction_coefficient=0.05):
        """
        Drive efficiency with the worm driving.

        ``eta = (cos(phi_n) - f tan(lambda)) / (cos(phi_n) + f / tan(lambda))``

        Args:
            friction_coefficient (float): Sliding friction coefficient
                (default: 0.05, typical for a lubricated steel worm on a
                bronze wheel).

        Returns:
            float: Efficiency (0 to 1).

        Raises:
            ValueError: If the friction coefficient is negative.
        """
        if friction_coefficient < 0:
            raise ValueError("Friction coefficient must be non-negative")
        phi_n = math.radians(self.pressure_angle)
        lam = math.radians(self.lead_angle)
        return ((math.cos(phi_n) - friction_coefficient * math.tan(lam))
                / (math.cos(phi_n) + friction_coefficient / math.tan(lam)))

    def is_self_locking(self, friction_coefficient=0.05):
        """
        Check whether the drive is self-locking (wheel cannot drive worm).

        Self-locking occurs approximately when the friction coefficient
        exceeds ``tan(lead angle) * cos(phi_n)``, i.e. when the lead
        angle is below the friction angle.

        Args:
            friction_coefficient (float): Sliding friction coefficient.

        Returns:
            bool: True if the drive is self-locking.
        """
        phi_n = math.radians(self.pressure_angle)
        lam = math.radians(self.lead_angle)
        return friction_coefficient >= math.tan(lam) * math.cos(phi_n)

    def permissible_load(self, wheel, wheel_material_key="bronze_chilled"):
        """
        Rough Buckingham permissible wear load on the wheel.

        ``W_allow = K * d_wheel * b`` with K a load-stress (wear) factor
        in N/mm^2. A coarse educational check for sizing, not an
        AGMA 6034 rating.

        Args:
            wheel (WormWheel): Mating worm wheel. Its ``face_width`` must
                be set.
            wheel_material_key (str): Key into
                :data:`mecapy.gears.agma_data.WORM_WEAR_K` (default:
                "bronze_chilled" — chill-cast bronze wheel on a hardened
                steel worm).

        Returns:
            float: Permissible tangential load on the wheel in N.

        Raises:
            ValueError: If the wheel face width is not set or the
                material key is unknown.
        """
        from .agma_data import WORM_WEAR_K
        from .transmission import _check_mesh

        _check_mesh(self, wheel)
        if wheel.face_width is None:
            raise ValueError("Wheel face width must be set for the wear check")
        if wheel_material_key not in WORM_WEAR_K:
            raise ValueError(
                f"Unknown wear factor key {wheel_material_key!r}; "
                f"available: {sorted(WORM_WEAR_K)}"
            )
        k = WORM_WEAR_K[wheel_material_key]
        return k * wheel.pitch_diameter * wheel.face_width

    def __repr__(self):
        return (
            f"Worm(starts={self.starts}, module={self.module}, "
            f"pitch_diameter={self.pitch_diameter}, "
            f"material={self.material!r})"
        )


class WormWheel(Gear):
    """
    Worm wheel (the gear member of a worm drive).

    Standard :class:`~mecapy.gears.Gear` geometry applies in the wheel's
    transverse plane; its module equals the worm's axial module for a
    90-degree drive. Classically made of bronze to pair with a hardened
    steel worm.

    Attributes:
        teeth (int): Number of teeth.
        module (float): Transverse module in mm.
        pressure_angle (float): Pressure angle in degrees.
        face_width (float): Face width in mm (None if not set).
        material (str): Material type (default: "bronze").
    """

    _min_teeth = 4

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 face_width=None, material="bronze", name=None,
                 diametral_pitch=None):
        """
        Initialize a WormWheel object.

        Args:
            teeth (int): Number of teeth (>= 4).
            module (float): Transverse module in mm (or give
                ``diametral_pitch``).
            pressure_angle (float): Pressure angle in degrees (default 20).
            face_width (float): Face width in mm (needed for the wear
                check).
            material (str): Material type (default: "bronze").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to module.
        """
        super().__init__(teeth, module=module, pressure_angle=pressure_angle,
                         material=material, name=name,
                         diametral_pitch=diametral_pitch)
        if face_width is not None and face_width <= 0:
            raise ValueError("Face width must be strictly positive")
        self.face_width = face_width
