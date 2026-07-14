"""Shaft design and analysis module."""

import math

from ..base import MechaElement


class Shaft(MechaElement):
    """
    Shaft design and analysis.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement` and adds a specialized torsional
    shear-stress calculation for solid circular shafts.

    Attributes:
        diameter (float): Shaft diameter in mm.
        length (float): Shaft length in mm.
        material (str): Material type.
    """

    LATEX_FIELDS = [
        ("diameter", "Diameter", "mm"),
        ("length", "Length", "mm"),
        ("polar_moment", "Polar moment $J$", "mm$^4$"),
    ]

    def __init__(self, diameter, length, material="steel", name=None):
        """
        Initialize a Shaft object.

        Args:
            diameter (float): Shaft diameter in mm.
            length (float): Shaft length in mm.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the shaft.
        """
        super().__init__(name=name, material=material)
        self.diameter = diameter
        self.length = length

    @property
    def polar_moment(self):
        """float: Polar second moment of area J = pi * d^4 / 32 (mm^4)."""
        return math.pi * self.diameter ** 4 / 32

    def torsional_stress(self, torque):
        """
        Calculate the maximum torsional shear stress for a solid shaft.

        Uses ``tau = T * r / J`` with ``r = d / 2`` and
        ``J = pi * d^4 / 32``.

        Args:
            torque (float): Applied torque (N*mm for a diameter in mm).

        Returns:
            float: Maximum surface shear stress (MPa for a diameter in mm
            and a torque in N*mm).
        """
        radius = self.diameter / 2
        return torque * radius / self.polar_moment

    def __repr__(self):
        return f"Shaft(diameter={self.diameter}mm, length={self.length}mm, material={self.material!r})"
