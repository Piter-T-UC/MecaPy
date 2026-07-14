"""Base class for all mechanical elements in MecaPy."""

from .materials import Material, get_material


class MechaElement:
    """
    Base class for every mechanical element.

    All components (beams, gears, shafts, bolts, ...) inherit from this
    class. It provides shared access to material properties and the common
    stress/safety-factor calculations that every element needs.

    Attributes:
        name (str): Optional identifier for the element.
        material: Material name (looked up in the database) or a
            :class:`~mecapy.materials.Material` instance.
    """

    def __init__(self, name=None, material="steel"):
        """
        Initialize a mechanical element.

        Args:
            name (str): Optional identifier for the element.
            material: Material name (default: "steel") that must exist in
                the material database, or a
                :class:`~mecapy.materials.Material` instance.
        """
        self.name = name
        self.material = material

    @property
    def material_properties(self):
        """Material: This element's material, resolved from the database.

        The returned :class:`~mecapy.materials.Material` also supports
        dictionary-style access (``material_properties["yield_strength"]``).
        """
        if isinstance(self.material, Material):
            return self.material
        return get_material(self.material)

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

    # ------------------------------------------------------------------
    # LaTeX / reporting
    # ------------------------------------------------------------------
    #: Per-class list of ``(attribute, label, unit)`` rows for LaTeX export.
    #: Subclasses override this to describe their defining quantities.
    LATEX_FIELDS = []

    @property
    def element_type(self):
        """str: The element's class name (e.g. "Beam", "Gear")."""
        return type(self).__name__

    def _material_name(self):
        if isinstance(self.material, Material):
            return self.material.name
        return self.material

    def latex_summary(self):
        """Return ``(quantity, value, unit)`` rows describing this element.

        Used by :class:`mecapy.report.Report` to build LaTeX tables. The
        base implementation lists the material plus every entry in
        :attr:`LATEX_FIELDS`; subclasses extend the class attribute rather
        than overriding this method.
        """
        rows = [("Material", self._material_name(), "")]
        for attr, label, unit in self.LATEX_FIELDS:
            try:
                value = getattr(self, attr)
            except Exception:
                continue
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            rows.append((label, value, unit))
        return rows

    def to_latex(self, heading="subsection"):
        """Return a LaTeX snippet (heading + property table) for this element."""
        from .report import render_element
        return render_element(self, heading=heading)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, material={self.material!r})"
