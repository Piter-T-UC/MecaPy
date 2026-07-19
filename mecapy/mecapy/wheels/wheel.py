"""Wheel design and analysis module."""

from ..base import MechaElement


class Wheel(MechaElement):
    """
    Wheel design and analysis.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`.

    Attributes:
        radius (float): Wheel radius in m.
        mass (float): Wheel mass in kg.
        material (str): Material type.
    """

    def __init__(self, radius, material="steel", mass=None, name=None):
        """
        Initialize a Wheel object.

        Args:
            radius (float): Wheel radius in m. Must be strictly positive.
            material (str): Material type (default: "steel").
            mass (float): Optional wheel mass in kg.
            name (str): Optional identifier for the wheel.

        Raises:
            ValueError: If ``radius`` is not strictly positive.
        """
        super().__init__(name=name, material=material)
        if radius <= 0:
            raise ValueError("Radius must be strictly positive")
        self.radius = radius
        self.mass = mass

    def __repr__(self):
        return f"Wheel(radius={self.radius}m, material={self.material!r}, mass={self.mass})"
