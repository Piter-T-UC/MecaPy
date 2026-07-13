"""Bolt design and analysis module."""

from ..base import MechaElement


class Bolt(MechaElement):
    """
    Bolt design and analysis.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`. The inherited
    :meth:`~mecapy.base.MechaElement.calculate_stress` gives the tensile
    stress in the bolt for a given axial force and stress area.

    Attributes:
        diameter (float): Bolt diameter in mm.
        length (float): Bolt length in mm.
        grade (str): Bolt grade (e.g., "M10").
        material (str): Material type.
    """

    def __init__(self, diameter, length, grade="M10", material="steel", name=None):
        """
        Initialize a Bolt object.

        Args:
            diameter (float): Bolt diameter in mm.
            length (float): Bolt length in mm.
            grade (str): Bolt grade (default: "M10").
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the bolt.
        """
        super().__init__(name=name, material=material)
        self.diameter = diameter
        self.length = length
        self.grade = grade

    def __repr__(self):
        return f"Bolt(grade={self.grade}, length={self.length}mm, material={self.material!r})"
