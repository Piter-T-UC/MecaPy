"""Beam analysis module."""


class Beam:
    """
    Base class for beam analysis and design.

    This class provides methods for analyzing and designing beams
    under various loading and support conditions.

    Attributes:
        length (float): Beam length in meters
        material (str): Material type
        section (dict): Cross-sectional properties
    """

    def __init__(self, length, material="steel", section=None):
        """
        Initialize a Beam object.

        Args:
            length (float): Beam length in meters
            material (str): Material type (default: "steel")
            section (dict): Cross-sectional properties (optional)
        """
        self.length = length
        self.material = material
        self.section = section or {}

    def __repr__(self):
        return f"Beam(length={self.length}m, material={self.material})"
