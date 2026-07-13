"""Shaft design and analysis module."""


class Shaft:
    """
    Base class for shaft design and analysis.

    This class provides methods for analyzing and designing
    rotating shafts in mechanical systems.

    Attributes:
        diameter (float): Shaft diameter in mm
        length (float): Shaft length in mm
        material (str): Material type
    """

    def __init__(self, diameter, length, material="steel"):
        """
        Initialize a Shaft object.

        Args:
            diameter (float): Shaft diameter in mm
            length (float): Shaft length in mm
            material (str): Material type (default: "steel")
        """
        self.diameter = diameter
        self.length = length
        self.material = material

    def __repr__(self):
        return f"Shaft(diameter={self.diameter}mm, length={self.length}mm, material={self.material})"
