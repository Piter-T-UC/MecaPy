"""Bolt design and analysis module.

Units convention: geometry in mm, forces in N, moments in N*mm and
stresses in MPa (N/mm^2), consistent with the other element modules
(gears, shafts). Metric and imperial bolts share this one class and this
one unit convention: Unified inch threads and SAE J429 grades are
converted to mm/MPa in :mod:`mecapy.bolts.thread_data`, so nothing
downstream needs to know which system a bolt came from.

Dimensional inputs additionally accept a ``pint.Quantity`` (e.g.
``length=2 * ureg.inch``), which is converted to mm at the boundary; plain
floats are assumed to be in mm and behave exactly as before.
"""

from math import pi

from ..base import MechaElement
from ..materials import CyclicLoadCases, Material
from ..utils.units import to_magnitude
from .thread_data import (
    SHIGLEY_MINOR_COEFF,
    SHIGLEY_PITCH_COEFF,
    get_property_class,
    get_thread,
    normalize_thread_size,
    shigley_thread_geometry,
)


class Bolt(CyclicLoadCases, MechaElement):
    """
    Bolt design and analysis (ISO metric or Unified inch).

    Thread geometry (nominal diameter, pitch, tensile stress area) is
    looked up from the ISO coarse-thread table or the Unified inch (UNC /
    UNF) table, and strength values from the ISO 898-1 property-class
    table or the SAE J429 grade table. Inherits shared material access
    from :class:`~mecapy.base.MechaElement`; the elastic modulus used for
    stiffness and elongation comes from the element material.

    Bolts outside both tables (fine metric series, custom pitches) can be
    built from an explicit ``diameter`` and ``pitch`` instead of a
    ``size``; their stress area is then computed with Shigley's formulas
    (mean of pitch and minor diameters). The same formula can be forced
    on a table size with ``stress_area_method="shigley"``:

        Bolt(size="M10", length=50)                     # table area
        Bolt(size="M10", length=50, stress_area_method="shigley")
        Bolt(length=50, diameter=12, pitch=1.25)        # fine thread
        Bolt(size="1/2-13", length=50, property_class="5")  # UNC, SAE 5

    Attributes:
        size (str): Thread designation, e.g. "M10" or "1/2-13" (custom
            bolts get a synthesized "M<d>x<p>" designation).
        length (float): Bolt length (grip length) in mm.
        property_class (str): ISO 898-1 property class (e.g. "8.8") or
            SAE J429 grade (e.g. "5").
        material (str): Material type (used for elastic modulus).
        stress_area_method (str): "table" or "shigley" — source of
            :attr:`stress_area`.
    """

    def __init__(self, size=None, length=None, property_class="8.8", material="steel",
                 name=None, diameter=None, pitch=None, stress_area_method=None):
        """
        Initialize a Bolt object.

        Args:
            size (str): ISO metric or Unified inch thread designation
                (e.g. "M10", "1/2-13"). Mutually exclusive with
                ``diameter``/``pitch``.
            length (float): Bolt length (grip length) in mm, or a
                pint.Quantity of length. Required.
            property_class (str): ISO 898-1 property class or SAE J429
                grade (default: "8.8").
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the bolt.
            diameter (float): Nominal thread diameter in mm (or a
                pint.Quantity) for a custom (non-table) bolt. Requires
                ``pitch``.
            pitch (float): Thread pitch in mm (or a pint.Quantity) for a
                custom bolt. Requires ``diameter``.
            stress_area_method (str): "table" (default for table sizes)
                or "shigley" to compute the stress area with Shigley's
                formulas. Custom bolts always use "shigley".

        Raises:
            ValueError: If the thread size or property class is unknown,
                the size/diameter/pitch combination is inconsistent, or
                ``length`` is missing or not strictly positive.
        """
        super().__init__(name=name, material=material)
        if stress_area_method not in (None, "table", "shigley"):
            raise ValueError(
                f"Unknown stress_area_method {stress_area_method!r}. "
                "Valid values: 'table', 'shigley'"
            )
        if size is not None:
            if diameter is not None or pitch is not None:
                raise ValueError(
                    "Provide either a table size or diameter and pitch, not both"
                )
            self._size = normalize_thread_size(size)  # validates (raises on unknown)
            self._custom_diameter = None
            self._custom_pitch = None
            self._stress_area_method = stress_area_method or "table"
        elif diameter is not None and pitch is not None:
            if stress_area_method == "table":
                raise ValueError(
                    "stress_area_method='table' requires a table size "
                    "(e.g. 'M10'), not a custom diameter and pitch"
                )
            diameter = to_magnitude(diameter, "mm")
            pitch = to_magnitude(pitch, "mm")
            shigley_thread_geometry(diameter, pitch)  # validate the geometry
            self._size = None
            self._custom_diameter = diameter
            self._custom_pitch = pitch
            self._stress_area_method = "shigley"
        elif diameter is not None or pitch is not None:
            raise ValueError("Custom bolts require both diameter and pitch")
        else:
            raise ValueError(
                "Provide a thread size (e.g. 'M10') or diameter and pitch"
            )
        if length is None:
            raise ValueError("Bolt length must be provided")
        self.length = length  # validating setter
        self.property_class = property_class  # validating setter

    # ---- Settable primary inputs (recompute the lookups, never cache) ----

    @property
    def length(self):
        """float: Bolt length (grip length) in mm. Settable (D1)."""
        return self._length

    @length.setter
    def length(self, value):
        value = to_magnitude(value, "mm")
        if value <= 0:
            raise ValueError("Bolt length must be strictly positive")
        self._length = value

    @property
    def size(self):
        """str: Thread designation, e.g. "M10" or "1/2-13" (synthesized
        "M<d>x<p>" for a custom bolt). Settable to another table size (D1)."""
        if self._size is not None:
            return self._size
        return f"M{self._custom_diameter:g}x{self._custom_pitch:g}"

    @size.setter
    def size(self, value):
        self._size = normalize_thread_size(value)  # validates the new size
        self._custom_diameter = None
        self._custom_pitch = None

    @property
    def stress_area_method(self):
        """str: Source of :attr:`stress_area` — "table" or "shigley"."""
        return self._stress_area_method

    @stress_area_method.setter
    def stress_area_method(self, value):
        if value not in ("table", "shigley"):
            raise ValueError("stress_area_method must be 'table' or 'shigley'")
        if value == "table" and self._size is None:
            raise ValueError(
                "stress_area_method='table' requires a table size, "
                "not a custom (diameter+pitch) bolt"
            )
        self._stress_area_method = value

    @property
    def property_class(self):
        """str: ISO 898-1 property class (e.g. "8.8") or SAE J429 grade
        (e.g. "5"). Settable (D1)."""
        return self._property_class

    @property_class.setter
    def property_class(self, value):
        get_property_class(value)  # validate the class (raises on unknown)
        self._property_class = value

    @property
    def _thread(self):
        """dict: Thread geometry resolved fresh from the current size (or the
        custom diameter/pitch), so mutating ``size`` updates every derived
        dimension — never cached."""
        if self._size is not None:
            thread = dict(get_thread(self._size))
            if self._stress_area_method == "shigley":
                thread["stress_area"] = shigley_thread_geometry(
                    thread["nominal_diameter"], thread["pitch"]
                )["stress_area"]
            return thread
        return shigley_thread_geometry(self._custom_diameter, self._custom_pitch)

    @property
    def _strength(self):
        """dict: Property-class (or SAE grade) strengths resolved fresh from
        :attr:`property_class` and the current nominal diameter, so mutating
        either updates the strengths — SAE grades 2 and 5 are derated above
        3/4 in and 1 in respectively."""
        return get_property_class(self._property_class, self.nominal_diameter)

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
        """float: Thread pitch in mm."""
        return self._thread["pitch"]

    @property
    def threads_per_inch(self):
        """float: Threads per inch, 25.4/p — the imperial read-back of
        :attr:`pitch` (mirrors ``Gear.diametral_pitch``)."""
        return 25.4 / self.pitch

    @property
    def thread_series(self):
        """str: Thread series — "ISO metric", "UNC", "UNF", or "custom" for a
        bolt built from an explicit diameter and pitch."""
        if self._size is None:
            return "custom"
        return self._thread.get("series", "ISO metric")

    @property
    def minor_diameter(self):
        """float: Minor (root) diameter d_r = d - 1.226869*p in mm (Shigley)."""
        return self.nominal_diameter - SHIGLEY_MINOR_COEFF * self.pitch

    @property
    def pitch_diameter(self):
        """float: Pitch diameter d_p = d - 0.649519*p in mm (Shigley)."""
        return self.nominal_diameter - SHIGLEY_PITCH_COEFF * self.pitch

    @property
    def stress_area(self):
        """float: Tensile stress area As in mm^2.

        This is the effective area resisting tension in the threaded
        portion, smaller than the nominal shank area. Source depends on
        :attr:`stress_area_method`: the ISO 898-1 table ("table") or
        Shigley's formula At = (pi/4)*((d_p + d_r)/2)^2 ("shigley").
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

    def _fatigue_material(self):
        """Material: bolt material carrying the property-class strengths.

        Overrides the :class:`~mecapy.materials.CyclicLoadCases` default so
        the Miner S-N curve uses this bolt's ISO 898-1 tensile/yield
        strengths (e.g. 800/640 MPa for class 8.8) rather than the generic
        database steel. The Marin factors keep their defaults (machined,
        kb = 1), so the endurance limit is an approximation — bolted-joint
        fatigue proper (preload, joint constant) is a separate analysis.
        """
        return Material(base_material=self.material,
                        ultimate_strength=self.tensile_strength,
                        yield_strength=self.yield_strength)

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
