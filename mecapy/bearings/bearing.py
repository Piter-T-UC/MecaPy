"""Bearing design and analysis module."""

from ..base import MechaElement


class Bearing(MechaElement):
    """
    Bearing design and analysis.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`.

    Attributes:
        bore_diameter (float): Bearing bore diameter in mm.
        outer_diameter (float): Bearing outer diameter in mm.
        width (float): Bearing width in mm.
        bearing_type (str): Type of bearing.
    """

    def __init__(self, bore_diameter, outer_diameter, width,
                 bearing_type="ball", material="steel", name=None):
        """
        Initialize a Bearing object.

        Args:
            bore_diameter (float): Bearing bore diameter in mm.
            outer_diameter (float): Bearing outer diameter in mm.
            width (float): Bearing width in mm.
            bearing_type (str): Type of bearing (default: "ball").
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the bearing.
        """
        super().__init__(name=name, material=material)
        self.bore_diameter = bore_diameter
        self.outer_diameter = outer_diameter
        self.width = width
        self.bearing_type = bearing_type

    def __repr__(self):
        return (
            f"Bearing({self.bore_diameter}/{self.outer_diameter}x{self.width}, "
            f"type={self.bearing_type})"
        )
