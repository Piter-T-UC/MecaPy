"""Gear design and analysis module."""


class Gear:
    """
    Base class for gear design and analysis.

    This class provides methods for designing and analyzing
    gears in mechanical transmissions.

    Attributes:
        teeth (int): Number of teeth
        module (float): Gear module
        material (str): Material type
    """

    def __init__(self, teeth, module, material="steel"):
        """
        Initialize a Gear object.

        Args:
            teeth (int): Number of teeth
            module (float): Gear module in mm
            material (str): Material type (default: "steel")
        """
        self.teeth = teeth
        self.module = module
        self.material = material

    def __repr__(self):
        return f"Gear(teeth={self.teeth}, module={self.module}, material={self.material})"
