"""Weld design and analysis module."""


class Weld:
    """
    Base class for weld design and analysis.

    This class provides methods for analyzing and designing welded
    connections in mechanical structures.

    Attributes:
        weld_type (str): Type of weld (e.g., "fillet", "butt")
        material (str): Weld material
        size (float): Weld size in mm
    """

    def __init__(self, weld_type, material="steel", size=None):
        """
        Initialize a Weld object.

        Args:
            weld_type (str): Type of weld (e.g., "fillet", "butt")
            material (str): Weld material (default: "steel")
            size (float): Weld size in mm (optional)
        """
        self.weld_type = weld_type
        self.material = material
        self.size = size

    def __repr__(self):
        return f"Weld(type={self.weld_type}, material={self.material}, size={self.size}mm)"
