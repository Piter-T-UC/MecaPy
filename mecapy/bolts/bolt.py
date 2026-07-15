"""Bolt design and analysis module.

Units convention: geometry in mm, forces in N, moments in N*mm and
stresses in MPa (N/mm^2), consistent with the other element modules
(gears, shafts). Bolt strength data comes from the ISO 898-1 property
classes in :mod:`mecapy.bolts.thread_data`.
"""

from math import pi

from ..base import MechaElement
from .thread_data import get_property_class, get_thread


class Bolt(MechaElement):
    """
    ISO metric bolt design and analysis.

    Thread geometry (nominal diameter, pitch, tensile stress area) is
    looked up from the ISO coarse-thread table, and strength values from
    the ISO 898-1 property-class table. Inherits shared material access
    from :class:`~mecapy.base.MechaElement`; the elastic modulus used for
    stiffness and elongation comes from the element material.

    Attributes:
        size (str): Thread designation, e.g. "M10".
        length (float): Bolt length (grip length) in mm.
        property_class (str): ISO 898-1 property class, e.g. "8.8".
        material (str): Material type (used for elastic modulus).
    """

    def __init__(self, size, length, property_class="8.8", material="steel", name=None):
        """
        Initialize a Bolt object.

        Args:
            size (str): ISO thread designation (e.g. "M10").
            length (float): Bolt length (grip length) in mm.
            property_class (str): ISO 898-1 property class (default: "8.8").
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the bolt.

        Raises:
            ValueError: If the thread size or property class is unknown,
                or if ``length`` is not strictly positive.
        """
        super().__init__(name=name, material=material)
        self._thread = get_thread(size)
        self._strength = get_property_class(property_class)
        if length <= 0:
            raise ValueError("Bolt length must be strictly positive")
        self.size = size
        self.length = length
        self.property_class = property_class

    # ---- Thread geometry ----

    @property
    def nominal_diameter(self):
        """float: Nominal (major) thread diameter in mm."""
        return self._thread["nominal_diameter"]

    @property
    def diameter(self):
        """float: Alias for :attr:`nominal_diameter` (mm)."""
        return self.nominal_diameter

    @property
    def pitch(self):
        """float: Thread pitch in mm (ISO coarse series)."""
        return self._thread["pitch"]

    @property
    def stress_area(self):
        """float: Tensile stress area As in mm^2 (ISO 898-1).

        This is the effective area resisting tension in the threaded
        portion, smaller than the nominal shank area.
        """
        return self._thread["stress_area"]

    @property
    def nominal_area(self):
        """float: Nominal shank area pi*d^2/4 in mm^2."""
        return pi * self.nominal_diameter ** 2 / 4

    # ---- Strength ----

    @property
    def proof_strength(self):
        """float: Proof strength of the property class in MPa."""
        return self._strength["proof_strength"]

    @property
    def tensile_strength(self):
        """float: Minimum tensile strength of the property class in MPa."""
        return self._strength["tensile_strength"]

    @property
    def yield_strength(self):
        """float: Minimum yield strength of the property class in MPa."""
        return self._strength["yield_strength"]

    @property
    def proof_load(self):
        """float: Proof load Fp = Sp * As in N."""
        return self.proof_strength * self.stress_area

    @property
    def recommended_preload(self):
        """float: Recommended preload in N.

        Uses the common rule of thumb Fi = 0.75 * Fp for reusable
        joints (use 0.90 * Fp for permanent joints).
        """
        return 0.75 * self.proof_load

    # ---- Stiffness and deformation ----

    @property
    def elastic_modulus(self):
        """float: Elastic modulus of the bolt material in MPa."""
        return self.material_properties["elastic_modulus"] / 1e6

    @property
    def stiffness(self):
        """float: Axial stiffness k = As * E / L in N/mm.

        Simple single-area model: the tensile stress area is assumed
        over the full length, which is conservative (a segmented
        shank/thread model would give a slightly stiffer bolt).
        """
        return self.stress_area * self.elastic_modulus / self.length

    def elongation(self, force):
        """
        Calculate the axial elongation under a tensile force.

        Uses delta = F * L / (As * E) = F / k.

        Args:
            force (float): Axial tensile force in N.

        Returns:
            float: Elongation in mm.
        """
        return force / self.stiffness

    # ---- Stress and safety ----

    def tensile_stress(self, force):
        """
        Calculate the tensile stress on the stress area.

        Args:
            force (float): Axial tensile force in N.

        Returns:
            float: Tensile stress in MPa (force / stress area).
        """
        return force / self.stress_area

    def bolt_safety_factor(self, force):
        """
        Calculate the safety factor against yielding of the bolt.

        Uses the property-class yield strength (ISO 898-1), unlike the
        inherited :meth:`~mecapy.base.MechaElement.safety_factor` which
        uses the generic material yield strength in Pa.

        Args:
            force (float): Axial tensile force in N.

        Returns:
            float: Ratio of property-class yield strength to the
            tensile stress on the stress area.

        Raises:
            ValueError: If ``force`` is zero.
        """
        if force == 0:
            raise ValueError("Force must be non-zero to compute a safety factor")
        return self.yield_strength / abs(self.tensile_stress(force))

    def __repr__(self):
        return (
            f"Bolt(size={self.size}, length={self.length}mm, "
            f"property_class={self.property_class}, material={self.material!r})"
        )
