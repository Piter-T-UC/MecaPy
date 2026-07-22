"""Base class for all mechanical elements in MecaPy."""

from .failure import n_coulomb_mohr
from .materials import Material, get_material_properties


class MechaElement:
    """
    Base class for every mechanical element.

    All components (beams, gears, shafts, bolts, ...) inherit from this
    class. It provides shared access to material properties and the common
    stress/safety-factor calculations that every element needs.

    Attributes:
        name (str): Optional identifier for the element.
        material (str or Material): Material name looked up in the material
            database, or a :class:`~mecapy.materials.Material` instance.
    """

    def __init__(self, name=None, material="steel"):
        """
        Initialize a mechanical element.

        Args:
            name (str): Optional identifier for the element.
            material (str or Material): Material name (default: "steel"),
                which must exist in the material database (see
                :mod:`mecapy.materials`), or a
                :class:`~mecapy.materials.Material` instance.
        """
        self.name = name
        self.material = material

    @property
    def material_properties(self):
        """dict: SI (Pa) properties of this element's material."""
        if isinstance(self.material, Material):
            return self.material.properties_si
        return get_material_properties(self.material)

    def calculate_stress(self, force, area):
        """
        Calculate the direct (axial) stress on the element.

        Uses the fundamental relation ``sigma = F / A``.

        Note: this base method is strict SI — ``force`` in N and ``area``
        in **m^2**, giving a stress in **Pa**. The geometry modules
        (bolts, gears, shafts, ...) work in mm/N/MPa, so pass an area in
        m^2 here (or use the module's own stress method) to avoid a
        unit mismatch.

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
        Static safety factor for a given stress, chosen by ductility.

        For a ductile material the ductile yield criterion applies,
        ``n = Sy / |sigma|``. For a material flagged ``brittle`` in the
        database (e.g. gray cast iron) the Coulomb-Mohr fracture criterion
        applies instead (Shigley Ch. 5): ``n = Sut / sigma`` in tension and
        ``n = Suc / |sigma|`` in compression — using Sy on a brittle
        material would be non-physical and can be unsafe.

        Args:
            stress (float): Applied uniaxial stress in Pascals (signed:
                positive tension, negative compression).

        Returns:
            float: Safety factor; > 1 means the element is safe.

        Raises:
            ValueError: If ``stress`` is zero.
        """
        if stress == 0:
            raise ValueError("Stress must be non-zero to compute a safety factor")
        props = self.material_properties
        if props.get("brittle", False):
            sut = props["ultimate_strength"]
            suc = props.get("ultimate_compressive_strength", sut)
            # Uniaxial Coulomb-Mohr: principals are (sigma, 0) in tension,
            # (0, sigma) in compression.
            s1, s3 = (stress, 0.0) if stress > 0 else (0.0, stress)
            return n_coulomb_mohr(s1, s3, sut, suc)
        return props["yield_strength"] / abs(stress)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, material={self.material!r})"
