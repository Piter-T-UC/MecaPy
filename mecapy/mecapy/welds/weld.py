"""Weld design and analysis module.

A :class:`Weld` is a single weld run laid along a geometric path (a
straight :class:`~mecapy.welds.geometry.WeldLine` or a full
:class:`~mecapy.welds.geometry.WeldCircle`), made with a given electrode.
It exposes the throat geometry and the *unit* second moments of area used
by the "weld as a line" method (Shigley's *Mechanical Engineering
Design*, Ch. 9). Groups of welds are analysed by
:class:`~mecapy.welds.welded_union.WeldedUnion`.

Units: geometry in mm, forces in N, moments in N*mm, stresses in MPa.
"""

from ..base import MechaElement
from .electrode_data import (
    AISC_SHEAR_ALLOWABLE_FACTOR,
    BASE_METAL_SHEAR_ALLOWABLE_FACTOR,
    get_electrode,
)
from .geometry import WeldPath

# Fillet-weld throat factor: throat = 0.707 * leg (45-degree fillet).
FILLET_THROAT_FACTOR = 0.707


class Weld(MechaElement):
    """
    Single weld run along a geometric path.

    Inherits shared material access from
    :class:`~mecapy.base.MechaElement`. The weld strength comes from the
    electrode (see :mod:`mecapy.welds.electrode_data`), not the base
    material.

    Attributes:
        path (WeldPath): The weld geometry (line or circle).
        weld_type (str): "fillet" (throat = 0.707*size) or "butt"
            (full-penetration, throat = size).
        size (float): Weld size in mm — the leg length for a fillet weld
            or the plate thickness for a butt weld (None if not set).
        electrode (str): Electrode designation, e.g. "E70xx".
        material (str): Base material type.
    """

    def __init__(self, path, size=None, weld_type="fillet",
                 electrode="E70xx", material="steel", name=None):
        """
        Initialize a Weld object.

        Args:
            path (WeldPath): Weld geometry — a
                :class:`~mecapy.welds.geometry.WeldLine` or
                :class:`~mecapy.welds.geometry.WeldCircle`.
            size (float): Weld size in mm (fillet leg or butt thickness).
                Optional for pure geometry; required to get a throat area
                or a stress.
            weld_type (str): "fillet" (default) or "butt".
            electrode (str): Electrode designation (default "E70xx").
            material (str): Base material type (default "steel").
            name (str): Optional identifier.

        Raises:
            ValueError: If ``path`` is not a WeldPath, ``weld_type`` is
                unknown, ``size`` is non-positive, or the electrode is
                unknown.
        """
        super().__init__(name=name, material=material)
        if not isinstance(path, WeldPath):
            raise ValueError("path must be a WeldLine or WeldCircle")
        if weld_type not in ("fillet", "butt"):
            raise ValueError("weld_type must be 'fillet' or 'butt'")
        if size is not None and size <= 0:
            raise ValueError("Weld size must be strictly positive")
        self.path = path
        self.weld_type = weld_type
        self.size = size
        self.electrode = electrode
        self._electrode = get_electrode(electrode)

    # ---- Geometry ----

    @property
    def length(self):
        """float: Weld length in mm."""
        return self.path.length

    @property
    def centroid(self):
        """tuple: Centroid (x, y) of the weld in mm."""
        return self.path.centroid

    @property
    def throat(self):
        """float: Throat thickness in mm.

        ``0.707 * size`` for a fillet weld, ``size`` for a butt weld.

        Raises:
            ValueError: If the weld size has not been set.
        """
        if self.size is None:
            raise ValueError("Weld size must be set to compute the throat")
        if self.weld_type == "fillet":
            return FILLET_THROAT_FACTOR * self.size
        return self.size

    @property
    def throat_area(self):
        """float: Throat area in mm^2 (throat * length)."""
        return self.throat * self.length

    def unit_inertia_about(self, cx, cy):
        """
        Unit second moments of area about an external point (cx, cy).

        Uses the parallel-axis theorem to shift this weld's own unit
        inertia to axes through ``(cx, cy)`` (typically the weld-group
        centroid):
        ``Ix = Ix_own + L*(yc - cy)^2``,
        ``Iy = Iy_own + L*(xc - cx)^2``.

        Args:
            cx (float): Reference x in mm.
            cy (float): Reference y in mm.

        Returns:
            tuple: ``(Ix, Iy)`` in mm^3 (weld treated as a line of unit
            throat).
        """
        ix_own, iy_own = self.path.unit_inertia()
        xc, yc = self.centroid
        length = self.length
        ix = ix_own + length * (yc - cy) ** 2
        iy = iy_own + length * (xc - cx) ** 2
        return (ix, iy)

    def sample_points(self, n):
        """Sample ``n`` points along the weld path (delegates to path)."""
        return self.path.sample_points(n)

    # ---- Strength ----

    @property
    def electrode_tensile(self):
        """float: Electrode tensile strength in MPa."""
        return self._electrode["tensile_strength"]

    @property
    def electrode_yield(self):
        """float: Electrode yield strength in MPa."""
        return self._electrode["yield_strength"]

    @property
    def allowable_stress(self):
        """float: AISC allowable shear on the throat in MPa.

        ``0.30 * electrode tensile strength`` (Shigley Table 9-4).
        """
        return AISC_SHEAR_ALLOWABLE_FACTOR * self.electrode_tensile

    @property
    def base_metal_allowable(self):
        """float: AISC allowable shear on the adjacent base metal in MPa.

        ``0.40 * base-metal yield strength`` (Shigley Table 9-5/9-6). This
        is the second limit a welded joint must satisfy, alongside the
        electrode :attr:`allowable_stress`; the attached member can shear
        through even when the weld metal is safe.
        """
        sy_base = self.material_properties["yield_strength"] / 1e6  # Pa -> MPa
        return BASE_METAL_SHEAR_ALLOWABLE_FACTOR * sy_base

    @property
    def base_metal_factor(self):
        """float: Throat-to-leg stress ratio, throat / size.

        The group stresses are computed on the throat; the base metal
        shears on the weld leg (= ``size``), a larger area for a fillet, so
        the base-metal stress is this fraction of the throat stress (0.707
        for a fillet, 1 for a butt weld).
        """
        return self.throat / self.size

    def __repr__(self):
        return (
            f"Weld(type={self.weld_type}, path={self.path!r}, "
            f"size={self.size}mm, electrode={self.electrode!r})"
        )
