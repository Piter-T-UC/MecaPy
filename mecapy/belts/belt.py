"""Belt drive design and analysis module.

Covers flat and V-belt open drives. Units are SI: lengths in meters,
velocities in m/s, tensions in N and power in W.
"""

import math

from ..base import MechaElement


class Belt(MechaElement):
    """
    Flat or V-belt drive.

    Provides the standard belt-drive relations: geometry (wrap angle and
    belt length), the capstan/Euler tension ratio, centrifugal tension and
    the transmitted power. Inherits from :class:`~mecapy.base.MechaElement`.

    Attributes:
        belt_type (str): "flat" or "v".
        friction (float): Coefficient of friction between belt and pulley.
        mass_per_length (float): Belt mass per unit length in kg/m.
        groove_angle (float): Included V-groove angle in degrees (V-belts).
        material (str): Belt material.
    """

    LATEX_FIELDS = [
        ("belt_type", "Belt type", ""),
        ("friction", "Friction coefficient $\\mu$", ""),
        ("mass_per_length", "Mass per length", "kg/m"),
        ("groove_angle", "Groove angle", "deg"),
    ]

    def __init__(self, belt_type="flat", friction=0.3, mass_per_length=0.0,
                 groove_angle=38.0, material="steel", name=None):
        """
        Initialize a Belt object.

        Args:
            belt_type (str): "flat" or "v" (default: "flat").
            friction (float): Belt-pulley coefficient of friction.
            mass_per_length (float): Belt mass per unit length in kg/m.
            groove_angle (float): Included V-groove angle in degrees, used
                for V-belts (default: 38).
            material (str): Belt material (default: "steel").
            name (str): Optional identifier for the belt.
        """
        super().__init__(name=name, material=material)
        self.belt_type = belt_type
        self.friction = friction
        self.mass_per_length = mass_per_length
        self.groove_angle = groove_angle

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @staticmethod
    def wrap_angle(large_diameter, small_diameter, center_distance):
        """
        Wrap (contact) angle on the small pulley for an open drive.

        Args:
            large_diameter (float): Diameter of the large pulley (m).
            small_diameter (float): Diameter of the small pulley (m).
            center_distance (float): Center-to-center distance (m).

        Returns:
            float: Wrap angle on the small pulley in radians (the smaller
            of the two, which governs slip).
        """
        return math.pi - 2 * math.asin(
            (large_diameter - small_diameter) / (2 * center_distance)
        )

    @staticmethod
    def belt_length(large_diameter, small_diameter, center_distance):
        """
        Length of an open belt drive.

        ``L = 2C + (pi/2)(D + d) + (D - d)^2 / (4C)``

        Args:
            large_diameter (float): Diameter of the large pulley (m).
            small_diameter (float): Diameter of the small pulley (m).
            center_distance (float): Center-to-center distance (m).

        Returns:
            float: Belt length in meters.
        """
        return (
            2 * center_distance
            + (math.pi / 2) * (large_diameter + small_diameter)
            + (large_diameter - small_diameter) ** 2 / (4 * center_distance)
        )

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------
    @property
    def _effective_friction_angle(self):
        """Effective friction exponent multiplier (accounts for V-groove)."""
        if self.belt_type == "v":
            return self.friction / math.sin(math.radians(self.groove_angle) / 2)
        return self.friction

    def tension_ratio(self, wrap_angle):
        """
        Ratio of tight-side to slack-side tension (Euler / capstan).

        Flat belt: ``T1/T2 = exp(mu * theta)``.
        V-belt:    ``T1/T2 = exp(mu * theta / sin(beta/2))``.

        Args:
            wrap_angle (float): Wrap angle on the driver pulley in radians.

        Returns:
            float: Tension ratio (T1 - Tc) / (T2 - Tc).
        """
        return math.exp(self._effective_friction_angle * wrap_angle)

    def centrifugal_tension(self, velocity):
        """
        Centrifugal tension ``Tc = m * v^2``.

        Args:
            velocity (float): Belt velocity in m/s.

        Returns:
            float: Centrifugal tension in N.
        """
        return self.mass_per_length * velocity ** 2

    def power(self, tight_tension, slack_tension, velocity):
        """
        Power transmitted by the belt.

        Args:
            tight_tension (float): Tight-side tension T1 in N.
            slack_tension (float): Slack-side tension T2 in N.
            velocity (float): Belt velocity in m/s.

        Returns:
            float: Transmitted power in W.
        """
        return (tight_tension - slack_tension) * velocity

    def max_power(self, max_tension, velocity, wrap_angle):
        """
        Maximum power for a given allowable tight-side tension.

        Accounts for centrifugal tension: only ``T1 - Tc`` and ``T2 - Tc``
        participate in the friction relation.

        Args:
            max_tension (float): Allowable tight-side tension T1 in N.
            velocity (float): Belt velocity in m/s.
            wrap_angle (float): Wrap angle on the driver pulley in radians.

        Returns:
            float: Maximum transmissible power in W.
        """
        Tc = self.centrifugal_tension(velocity)
        ratio = self.tension_ratio(wrap_angle)
        slack = Tc + (max_tension - Tc) / ratio
        return (max_tension - slack) * velocity

    def __repr__(self):
        return f"Belt(type={self.belt_type!r}, friction={self.friction})"
