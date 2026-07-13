"""Bolt design and analysis module."""


class Bolt:
    """
    Base class for bolt design and analysis.

    This class provides methods for designing and analyzing bolted
    connections in mechanical applications.

    Attributes:
        diameter (float): Bolt diameter in mm
        length (float): Bolt length in mm
        grade (str): Bolt grade (e.g., "M10x1.5")
        material (str): Material type
    """

    def __init__(self, diameter, length, grade="M10", material="steel"):
        """
        Initialize a Bolt object.

        Args:
            diameter (float): Bolt diameter in mm
            length (float): Bolt length in mm
            grade (str): Bolt grade (default: "M10")
            material (str): Material type (default: "steel")
        """
        self.diameter = diameter
        self.length = length
        self.grade = grade
        self.material = material

    def __repr__(self):
        return f"Bolt(grade={self.grade}, length={self.length}mm, material={self.material})"
