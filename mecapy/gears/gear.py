"""Gear design and analysis module."""

from ..base import MechaElement


class Gear(MechaElement):
    """
    Gear design and analysis.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`.

    Attributes:
        teeth (int): Number of teeth.
        module (float): Gear module.
        material (str): Material type.
    """

    def __init__(self, teeth, module, material="steel", name=None):
        """
        Initialize a Gear object.

        Args:
            teeth (int): Number of teeth.
            module (float): Gear module in mm.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the gear.
        """
        super().__init__(name=name, material=material)
        self.teeth = teeth
        self.module = module

    @property
    def pitch_diameter(self):
        """float: Pitch diameter (teeth * module)."""
        return self.teeth * self.module

    def __repr__(self):
        return f"Gear(teeth={self.teeth}, module={self.module}, material={self.material!r})"
