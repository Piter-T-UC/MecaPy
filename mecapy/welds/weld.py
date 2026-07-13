"""Weld design and analysis module."""

from ..base import MechaElement


class Weld(MechaElement):
    """
    Weld design and analysis.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`.

    Attributes:
        weld_type (str): Type of weld (e.g., "fillet", "butt").
        material (str): Weld material.
        size (float): Weld size (leg length / throat) in mm.
    """

    def __init__(self, weld_type, material="steel", size=None, name=None):
        """
        Initialize a Weld object.

        Args:
            weld_type (str): Type of weld (e.g., "fillet", "butt").
            material (str): Weld material (default: "steel").
            size (float): Weld size in mm (optional).
            name (str): Optional identifier for the weld.
        """
        super().__init__(name=name, material=material)
        self.weld_type = weld_type
        self.size = size

    def __repr__(self):
        return f"Weld(type={self.weld_type}, material={self.material!r}, size={self.size}mm)"
