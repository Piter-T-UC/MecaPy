"""Base class for all mechanical elements in MecaPy."""

from .materials import get_material_properties


class MechaElement:
    """
    Base class for every mechanical element.

    All components (beams, gears, shafts, bolts, ...) inherit from this
    class. It provides shared access to material properties and the common
    stress/safety-factor calculations that every element needs.

    Attributes:
        name (str): Optional identifier for the element.
        material (str): Material name, looked up in the material database.
    """

    def __init__(self, name=None, material="steel"):
        """
        Initialize a mechanical element.

        Args:
            name (str): Optional identifier for the element.
            material (str): Material name (default: "steel"). Must exist in
                the material database (see :mod:`mecapy.materials`).
        """
        self.name = name
        self.material = material

    @property
    def material_properties(self):
        """dict: Properties of this element's material from the database."""
        return get_material_properties(self.material)

    def calculate_stress(self, force, area):
        """
        Calculate the direct (axial) stress on the element.

        Uses the fundamental relation ``sigma = F / A``.

        Args:
            force (float): Applied force in Newtons.
            area (float): Cross-sectional area in m^2.

        Returns:
            float: Normal stress in Pascals.

        Raises:
            ValueError: If ``area`` is not strictly positive.
        """
        if area <= 0:
            raise ValueError("Area must be strictly positive")
        return force / area

    def safety_factor(self, stress):
        """
        Calculate the safety factor against yielding for a given stress.

        Args:
            stress (float): Applied stress in Pascals.

        Returns:
            float: Ratio of the material yield strength to the applied
            stress. Values greater than 1 indicate the element does not
            yield.

        Raises:
            ValueError: If ``stress`` is zero.
        """
        if stress == 0:
            raise ValueError("Stress must be non-zero to compute a safety factor")
        yield_strength = self.material_properties["yield_strength"]
        return yield_strength / abs(stress)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, material={self.material!r})"
