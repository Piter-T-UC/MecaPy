"""Bearing design and analysis module."""


class Bearing:
    """
    Base class for bearing design and analysis.

    This class provides methods for analyzing and selecting bearings
    for mechanical applications.

    Attributes:
        bore_diameter (float): Bearing bore diameter in mm
        outer_diameter (float): Bearing outer diameter in mm
        width (float): Bearing width in mm
        bearing_type (str): Type of bearing
    """

    def __init__(self, bore_diameter, outer_diameter, width, bearing_type="ball"):
        """
        Initialize a Bearing object.

        Args:
            bore_diameter (float): Bearing bore diameter in mm
            outer_diameter (float): Bearing outer diameter in mm
            width (float): Bearing width in mm
            bearing_type (str): Type of bearing (default: "ball")
        """
        self.bore_diameter = bore_diameter
        self.outer_diameter = outer_diameter
        self.width = width
        self.bearing_type = bearing_type

    def __repr__(self):
        return f"Bearing({self.bore_diameter}/{self.outer_diameter}x{self.width}, type={self.bearing_type})"
