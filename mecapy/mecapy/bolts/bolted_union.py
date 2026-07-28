"""Bolted union (bolt group) analysis module.

Distributes external loads over a pattern of identical bolts using the
elastic (rigid-plate) method. Units: coordinates in mm, forces in N,
moments in N*mm, stresses in MPa.

Dimensional inputs additionally accept a ``pint.Quantity`` (e.g.
``forces=(ureg.Quantity(2.5, "kN"), 0, 0)`` or a plate thickness in
inches), which is converted to the documented unit and unwrapped at the
boundary -- see :mod:`mecapy.utils.units`. Stored attributes and every
derived property remain plain floats.

Prefer ``ureg.Quantity(2.5, "kN")`` over the ``2.5 * ureg.kN`` idiom.
They are the same object at runtime, but pint builds its operator
overloads dynamically, so a type checker infers ``2.5 * ureg.kN`` as a
``Unit`` rather than a ``Quantity`` and flags any later attribute access
(``Cannot access attribute "to" for class "Unit"``).

Load convention (right-handed axes, z out of the joint plane):
    - forces = (Fx, Fy, Fz): Fx, Fy are in-plane shear on the group,
      Fz is the axial (out-of-plane) force.
    - moments = (Mx, My, Mz): Mx and My are bending moments about the
      x and y axes, Mz is the torsion moment about z.
    - All loads act at the centroid of the bolt group.

Bending reference: by default the bending moments Mx and My are resolved
about the *extreme* bolt on the compression side (the pivot), not about
the centroid -- a clamped plate cannot pull, so the joint rotates about
that bolt and every bolt sees tension. Torsion Mz always uses the
centroid. See :attr:`BoltedUnion.bending_reference`.
"""

from math import copysign, cos, hypot, log, pi, radians, sin, sqrt, tan

from ..base import MechaElement
from ..materials import get_material_properties
from ..utils.units import to_magnitude
from .bolt import Bolt
from .thread_data import ISO_COARSE_THREADS, UNIFIED_THREADS, get_property_class

# Default interface friction coefficient (dry steel on steel), matching
# the convention in mecapy.shafts.power_screw.
DEFAULT_MU = 0.3

# Washer-face (bolt head / nut bearing) diameter as a multiple of the
# nominal bolt diameter, dw = 1.5*d. It is both where Shigley's frustum
# cones start and the annulus the head presses on, so the member
# stiffness and the washer-pressure check share this one constant.
WASHER_FACE_RATIO = 1.5

# Distortion-energy shear yield, Ssy = 0.577*Sy -- the same convention as
# mecapy.joints (key, pin, rivet) and mecapy.couplings.flange.
SHEAR_YIELD_FACTOR = 0.577


def circular_pattern(n_bolts, radius):
    """
    Build a circular bolt pattern, evenly spaced starting at angle 0.

    Args:
        n_bolts (int): Number of bolts around the circle.
        radius (float): Pattern radius in mm (or a pint.Quantity of
            length).

    Returns:
        list: ``[bolt_number, x, y]`` rows (mm) suitable for
        :class:`BoltedUnion`'s ``positions``, numbered 1..n_bolts.

    Raises:
        ValueError: If ``n_bolts`` is not a strictly positive integer or
            ``radius`` is not strictly positive.
    """
    if n_bolts <= 0:
        raise ValueError("n_bolts must be strictly positive")
    radius = to_magnitude(radius, "mm")
    if radius <= 0:
        raise ValueError("radius must be strictly positive")
    return [
        [i + 1, radius * cos(2 * pi * i / n_bolts), radius * sin(2 * pi * i / n_bolts)]
        for i in range(n_bolts)
    ]


class BoltedUnion(MechaElement):
    """
    Bolted union (bolt group) under combined loading.

    All bolts are identical (same :class:`Bolt` instance), so the
    elastic distribution depends only on the bolt positions: direct
    shear and axial force split equally, torsion produces shear
    proportional to the distance from the centroid, and bending
    produces axial force proportional to the distance from the bending
    reference line -- by default the extreme (farthest) bolt on the
    compression side, see :attr:`bending_reference`.

    A clamped-member model can be attached with ``plates``: the stack of
    plates the bolts squeeze, each with its own thickness and material.
    Member stiffness then follows Shigley's 30-degree frustum-cone
    method, giving the joint constant C = kb / (kb + km) used by the
    preload-aware checks (:meth:`bolt_tensions`,
    :meth:`proof_safety_factors`, :meth:`separation_safety_factors`,
    :meth:`slip_safety_factors`) and by :meth:`minimum_bolt`.

    Attributes:
        bolt (Bolt): The bolt used at every position.
        positions (list): List of ``[bolt_number, x, y]`` rows in mm.
        forces (tuple): Applied forces (Fx, Fy, Fz) in N.
        moments (tuple): Applied moments (Mx, My, Mz) in N*mm.
        plates (list): Clamped stack as ``[(thickness, material), ...]``
            from bolt-head side to nut side, or None.
        tapped (bool): True when the last member is tapped and the bolt
            screws into it instead of taking a nut. The stiffness chain
            then uses :attr:`effective_grip` instead of :attr:`grip`.
        preload (float): Explicit per-bolt preload Fi in N, or None to
            use ``bolt.recommended_preload``.
        bending_reference (str): ``"extreme"`` (default) or
            ``"centroid"`` -- which line Mx and My are resolved about.
        cone_angle_deg (float): Half-angle of the Shigley frustum cones
            in degrees (default: 30). Assign directly (a number of
            degrees or a pint.Quantity angle) to use a different angle in
            :attr:`member_stiffness` and :attr:`joint_constant`.
    """

    def __init__(self, bolt, positions, forces=(0, 0, 0), moments=(0, 0, 0),
                 bending_reference="extreme", plates=None, preload=None,
                 tapped=False, name=None):
        """
        Initialize a bolted union.

        Args:
            bolt (Bolt): Bolt instance used at every position.
            positions (list): List of ``[bolt_number, x, y]`` rows,
                coordinates in mm (or pint.Quantity lengths).
            forces (tuple): Applied forces (Fx, Fy, Fz) in N (or
                pint.Quantity forces), acting at the bolt-group centroid
                (default: no force).
            moments (tuple): Applied moments (Mx, My, Mz) in N*mm (or
                pint.Quantity torques) about the centroid: Mx, My
                bending, Mz torsion (default: no moment).
            bending_reference (str): Line the bending moments Mx and My
                are resolved about: ``"extreme"`` (default) pivots on
                the farthest bolt on the compression side (tension-only
                bolt loads), ``"centroid"`` uses the centroidal axis.
                Torsion Mz always uses the centroid.
            plates (list): Optional clamped stack as
                ``[(thickness, material), ...]`` (mm or a pint.Quantity
                length, material name) from bolt-head side to nut side.
                At least two plates; with ``tapped`` the last one is the
                tapped member.
            preload (float): Optional per-bolt preload Fi in N (or a
                pint.Quantity force). Default None uses
                ``bolt.recommended_preload`` where needed.
            tapped (bool): True when the bolt screws into the last
                member instead of taking a nut (default: False). The
                grip used by the stiffness chain becomes
                :attr:`effective_grip`.
            name (str): Optional identifier for the union.

        Raises:
            ValueError: If ``bolt`` is not a Bolt, a position row does
                not have exactly 3 entries, bolt numbers repeat, the
                force/moment vectors do not have 3 components,
                ``bending_reference`` is unknown, or the plates/preload
                are invalid.
        """
        if not isinstance(bolt, Bolt):
            raise ValueError("bolt must be a Bolt instance")
        super().__init__(name=name, material=bolt.material)
        if not positions:
            raise ValueError("positions must contain at least one bolt")
        for row in positions:
            if len(row) != 3:
                raise ValueError(
                    f"Each position must be [bolt_number, x, y]; got {list(row)!r}"
                )
        numbers = [row[0] for row in positions]
        if len(set(numbers)) != len(numbers):
            raise ValueError("Bolt numbers must be unique")
        if len(forces) != 3:
            raise ValueError("forces must have 3 components (Fx, Fy, Fz)")
        if len(moments) != 3:
            raise ValueError("moments must have 3 components (Mx, My, Mz)")
        self.bolt = bolt
        self.positions = [
            [row[0], to_magnitude(row[1], "mm"), to_magnitude(row[2], "mm")]
            for row in positions
        ]
        self.forces = tuple(to_magnitude(f, "N") for f in forces)
        self.moments = tuple(to_magnitude(m, "N*mm") for m in moments)
        self.plates = None
        self.preload = None
        self._tapped = False
        self._hole_diameter = None
        self._centroid_override = None
        self._cone_angle_deg = 30.0
        self.bending_reference = bending_reference
        self.set_plates(plates, tapped)
        self.set_preload(preload)

    # ---- Bolt, plates and preload configuration ----

    def set_bolt(self, bolt):
        """
        Replace the union's bolt, keeping positions, loads, plates and preload.

        Args:
            bolt (Bolt): New bolt instance to use at every position.

        Raises:
            ValueError: If ``bolt`` is not a Bolt, is shorter than the
                grip defined by the plates, or the explicit preload
                exceeds the new bolt's proof load.
        """
        if not isinstance(bolt, Bolt):
            raise ValueError("bolt must be a Bolt instance")
        if self.plates is not None:
            self._validate_bolt_length(bolt, self.plates, self.tapped)
        if self.preload is not None and self.preload > bolt.proof_load:
            raise ValueError(
                f"Preload {self.preload} N exceeds the proof load "
                f"{bolt.proof_load:.0f} N of the new bolt"
            )
        self.bolt = bolt
        self.material = bolt.material

    def _validate_bolt_length(self, bolt, plates, tapped):
        """Check the bolt reaches through (or into) the member stack.

        A through-bolted joint just needs to span the grip. A tapped
        joint must reach past the members above the tapped one and must
        not bottom out in the blind hole.

        Raises:
            ValueError: If the bolt is too short, or too long for a
                tapped stack.
        """
        if not tapped:
            grip = sum(t for t, _ in plates)
            if bolt.length < grip:
                raise ValueError(
                    f"Bolt length {bolt.length} mm is shorter than the "
                    f"grip {grip} mm defined by the plates"
                )
            return
        upper = sum(t for t, _ in plates[:-1])
        tapped_thickness = plates[-1][0]
        if bolt.length <= upper:
            raise ValueError(
                f"Bolt length {bolt.length} mm does not reach into the "
                f"tapped member (the members above it are {upper} mm)"
            )
        if bolt.length > upper + tapped_thickness:
            raise ValueError(
                f"Bolt length {bolt.length} mm exceeds the tapped stack "
                f"{upper + tapped_thickness} mm: it would bottom out in "
                f"the blind hole"
            )

    def set_plates(self, plates, tapped=False):
        """
        Define (or clear) the clamped plate stack.

        Args:
            plates (list): ``[(thickness, material), ...]`` from
                bolt-head side to nut side, thickness in mm (or a
                pint.Quantity length) and material a name from the
                material database. Pass None to remove the member model.
            tapped (bool): True when the last row is a tapped member the
                bolt screws into rather than a plate held by a nut
                (default: False).

        Raises:
            ValueError: If a row is malformed, a thickness is not
                strictly positive, a material is unknown, fewer than two
                members are given, or the bolt does not suit the stack
                (too short, or bottoming out in a blind hole).
        """
        if plates is None:
            self.plates = None
            self._tapped = False
            return
        normalized = []
        for row in plates:
            if len(row) != 2:
                raise ValueError(
                    f"Each plate must be (thickness, material); got {list(row)!r}"
                )
            thickness, material = row
            thickness = to_magnitude(thickness, "mm")
            if thickness <= 0:
                raise ValueError("Plate thickness must be strictly positive")
            get_material_properties(material)  # raises for unknown materials
            normalized.append((float(thickness), material))
        if len(normalized) < 2:
            raise ValueError(
                "A joint clamps at least two members (for a tapped joint, "
                "the last one is the tapped member)"
            )
        self._validate_bolt_length(self.bolt, normalized, bool(tapped))
        self.plates = normalized
        self._tapped = bool(tapped)

    def set_preload(self, preload):
        """
        Set (or clear) the explicit per-bolt preload.

        Args:
            preload (float): Preload Fi in N (or a pint.Quantity force),
                or None to fall back to ``bolt.recommended_preload``.

        Raises:
            ValueError: If ``preload`` is not strictly positive or
                exceeds the bolt's proof load.
        """
        if preload is None:
            self.preload = None
            return
        preload = to_magnitude(preload, "N")
        if preload <= 0:
            raise ValueError("Preload must be strictly positive")
        if preload > self.bolt.proof_load:
            raise ValueError(
                f"Preload {preload} N exceeds the bolt proof load "
                f"{self.bolt.proof_load:.0f} N"
            )
        self.preload = float(preload)

    @property
    def cone_angle_deg(self):
        """float: Half-angle of the Shigley frustum cones, in degrees.

        Assign a plain number of degrees or a pint.Quantity angle; the
        stored value is always a plain float in degrees.
        """
        return self._cone_angle_deg

    @cone_angle_deg.setter
    def cone_angle_deg(self, angle):
        self._cone_angle_deg = to_magnitude(angle, "degree")

    @property
    def bending_reference(self):
        """str: Line the bending moments Mx and My are resolved about.

        ``"extreme"`` (default) pivots on the farthest bolt on the
        compression side: the clamped plate cannot pull, so the joint
        rotates about that bolt and every bolt takes tension, the
        compression being reacted by plate bearing at the pivot.
        ``"centroid"`` uses the centroidal axis, which splits the group
        into a tension half and a compression half.

        Torsion Mz always uses the centroid, in both modes.
        """
        return self._bending_reference

    @bending_reference.setter
    def bending_reference(self, mode):
        if mode not in ("extreme", "centroid"):
            raise ValueError(
                f"bending_reference must be 'extreme' or 'centroid'; got {mode!r}"
            )
        self._bending_reference = mode

    # ---- Geometry ----

    @property
    def n_bolts(self):
        """int: Number of bolts in the group."""
        return len(self.positions)

    @property
    def centroid(self):
        """tuple: Centroid (x, y) of the bolt group in mm.

        Simple average of the positions — all bolts have equal area.
        Assign to this property to override the analysis point (e.g. a
        known center of rigidity); assign None to fall back to the
        computed average again.
        """
        if self._centroid_override is not None:
            return self._centroid_override
        n = self.n_bolts
        x_bar = sum(row[1] for row in self.positions) / n
        y_bar = sum(row[2] for row in self.positions) / n
        return (x_bar, y_bar)

    @centroid.setter
    def centroid(self, point):
        """
        Override the point loads/moments are applied about and shear is
        measured from, instead of the computed geometric average.

        Args:
            point (tuple): (x, y) in mm (or pint.Quantity lengths), or
                None to remove the override and go back to the computed
                centroid.

        Raises:
            ValueError: If ``point`` is not None and does not have
                exactly 2 numeric components.
        """
        if point is None:
            self._centroid_override = None
            return
        if len(point) != 2:
            raise ValueError(f"centroid must be (x, y); got {list(point)!r}")
        self._centroid_override = (
            to_magnitude(point[0], "mm"),
            to_magnitude(point[1], "mm"),
        )

    def _relative_coords(self):
        """Return [(number, dx, dy)] with coordinates relative to the centroid."""
        x_bar, y_bar = self.centroid
        return [(row[0], row[1] - x_bar, row[2] - y_bar) for row in self.positions]

    # ---- Load distribution ----

    def _bending_axials(self, coords, moment, axis_label):
        """
        Per-bolt axial load from one bending moment.

        Args:
            coords (list): Coordinate of each bolt along the direction
                the moment bends in (dy for Mx, dx for My), in mm and in
                the same order as :attr:`positions`.
            moment (float): Signed moment in N*mm, oriented so that a
                positive value puts the bolts at high ``coords`` in
                tension (that is Mx for the dy coordinates and -My for
                the dx ones, following the sign convention of
                :meth:`bolt_forces`).
            axis_label (str): ``"Mx"`` or ``"My"``, for the error message.

        Returns:
            tuple: ``(axials, pivot_index)`` -- the per-bolt axial loads
            in N, and the index into ``coords`` of the pivot bolt
            (None in ``"centroid"`` mode or for a zero moment).

        Raises:
            ValueError: If the moment is non-zero but every bolt sits at
                the same coordinate, leaving no lever arm to resist it.
        """
        if moment == 0:
            return [0.0] * len(coords), None
        if self.bending_reference == "centroid":
            sum_c2 = sum(c ** 2 for c in coords)
            if sum_c2 == 0:
                raise ValueError(
                    f"Bending {axis_label} applied but all bolts lie on the "
                    f"same line: no lever arm to resist it"
                )
            return [moment * c / sum_c2 for c in coords], None
        # Extreme-bolt pivot: the plate cannot pull, so the joint rotates
        # about the farthest bolt on the compression side and every lever
        # arm is measured from there -- no bolt ends up in compression.
        sign = 1.0 if moment > 0 else -1.0
        signed = [sign * c for c in coords]
        pivot_index = min(range(len(signed)), key=lambda i: signed[i])
        arms = [s - signed[pivot_index] for s in signed]
        sum_d2 = sum(d ** 2 for d in arms)
        if sum_d2 == 0:
            raise ValueError(
                f"Bending {axis_label} applied but all bolts lie on the "
                f"same line: no lever arm to resist it"
            )
        return [abs(moment) * d / sum_d2 for d in arms], pivot_index

    @property
    def bending_pivots(self):
        """dict: Bolt number the joint pivots about for each bending moment.

        ``{"x": number_or_None, "y": number_or_None}`` where ``"x"`` is
        the pivot for Mx and ``"y"`` the pivot for My. A value is None
        when that moment is zero or when :attr:`bending_reference` is
        ``"centroid"`` (which has no pivot -- it rotates about the
        centroidal axis).

        Raises:
            ValueError: If a bending moment has no lever arm to resist it.
        """
        rel = self._relative_coords()
        numbers = [number for number, _dx, _dy in rel]
        mx, my = self.moments[0], self.moments[1]
        _, pivot_x = self._bending_axials([dy for _n, _dx, dy in rel], mx, "Mx")
        _, pivot_y = self._bending_axials([dx for _n, dx, _dy in rel], -my, "My")
        return {
            "x": numbers[pivot_x] if pivot_x is not None else None,
            "y": numbers[pivot_y] if pivot_y is not None else None,
        }

    def bolt_forces(self):
        """
        Distribute the applied loads over the bolts (elastic method).

        Per bolt:
            - direct shear: (Fx/n, Fy/n);
            - torsion Mz: shear of magnitude Mz*r_i / sum(r^2),
              perpendicular to the radius from the centroid (always
              measured from the centroid -- the group really does twist
              about it);
            - direct axial: Fz/n;
            - bending Mx, My: axial load proportional to the distance
              from the bending reference line, see
              :attr:`bending_reference`. With the default ``"extreme"``
              the lever arms are measured from the farthest bolt on the
              compression side, so every bolt takes tension (>= 0, zero
              at the pivot bolt) and the balancing compression is
              reacted by plate bearing at that pivot: the bending part
              of the axial loads no longer sums to zero, but their
              moment about the pivot line equals the applied moment.
              With ``"centroid"`` the arms are measured from the
              centroidal axis: Mx*dy_i / sum(dy^2) and
              -My*dx_i / sum(dx^2), half the bolts in compression.

        Sign convention in both modes: positive Mx puts the bolts at
        positive y in tension, positive My those at negative x.

        Returns:
            dict: ``{bolt_number: {"shear": (Fsx, Fsy),
            "shear_magnitude": float, "axial": float,
            "shear_direct": (Vx, Vy), "shear_torsion": (Tx, Ty),
            "axial_direct": float, "axial_bending_x": float,
            "axial_bending_y": float}}`` with forces in N. ``shear`` is
            the component-wise sum of ``shear_direct`` (from Fx, Fy) and
            ``shear_torsion`` (from Mz); ``axial`` is the sum of
            ``axial_direct`` (from Fz), ``axial_bending_x`` (from Mx)
            and ``axial_bending_y`` (from My).

        Raises:
            ValueError: If a torsion/bending moment is non-zero but the
                bolt pattern has no lever arm to resist it (all bolts at
                the centroid or on a single line).
        """
        fx, fy, fz = self.forces
        mx, my, mz = self.moments
        n = self.n_bolts
        rel = self._relative_coords()

        sum_r2 = sum(dx ** 2 + dy ** 2 for _, dx, dy in rel)
        if mz != 0 and sum_r2 == 0:
            raise ValueError("Torsion applied but all bolts are at the centroid")
        bending_x, _ = self._bending_axials([dy for _n, _dx, dy in rel], mx, "Mx")
        bending_y, _ = self._bending_axials([dx for _n, dx, _dy in rel], -my, "My")

        result = {}
        for index, (number, dx, dy) in enumerate(rel):
            vx = fx / n
            vy = fy / n
            if mz != 0:
                tx = -mz * dy / sum_r2
                ty = mz * dx / sum_r2
            else:
                tx = 0.0
                ty = 0.0
            fsx = vx + tx
            fsy = vy + ty
            axial_direct = fz / n
            axial = axial_direct + bending_x[index] + bending_y[index]
            result[number] = {
                "shear": (fsx, fsy),
                "shear_magnitude": sqrt(fsx ** 2 + fsy ** 2),
                "axial": axial,
                "shear_direct": (vx, vy),
                "shear_torsion": (tx, ty),
                "axial_direct": axial_direct,
                "axial_bending_x": bending_x[index],
                "axial_bending_y": bending_y[index],
            }
        return result

    def max_loaded_bolt(self):
        """
        Find the most loaded bolt (largest von Mises equivalent stress).

        Returns:
            tuple: ``(bolt_number, forces_dict)`` where ``forces_dict``
            is that bolt's entry from :meth:`bolt_forces`.
        """
        forces = self.bolt_forces()
        area = self.bolt.stress_area

        def equivalent(entry):
            sigma = entry["axial"] / area
            tau = entry["shear_magnitude"] / area
            return sqrt(sigma ** 2 + 3 * tau ** 2)

        number = max(forces, key=lambda k: equivalent(forces[k]))
        return number, forces[number]

    def safety_factors(self):
        """
        Calculate the safety factor against yielding for each bolt.

        Combines the axial stress and the shear stress on the tensile
        stress area with the von Mises criterion
        sigma_eq = sqrt(sigma^2 + 3*tau^2) — a simple single-plane
        model — and compares it with the property-class yield strength.
        Preload is not considered; for preloaded joints with a member
        model see :meth:`proof_safety_factors`,
        :meth:`separation_safety_factors` and
        :meth:`slip_safety_factors`.

        Returns:
            dict: ``{bolt_number: safety_factor}``. A bolt with no load
            gets ``float("inf")``.
        """
        area = self.bolt.stress_area
        yield_strength = self.bolt.yield_strength
        result = {}
        for number, entry in self.bolt_forces().items():
            sigma = entry["axial"] / area
            tau = entry["shear_magnitude"] / area
            sigma_eq = sqrt(sigma ** 2 + 3 * tau ** 2)
            result[number] = yield_strength / sigma_eq if sigma_eq > 0 else float("inf")
        return result

    # ---- Joint stiffness (clamped members) ----

    def _require_plates(self):
        """Raise if no member model has been defined."""
        if self.plates is None:
            raise ValueError(
                "No plates defined for this union; pass "
                "plates=[(thickness, material), ...] or call set_plates()"
            )

    @property
    def tapped(self):
        """bool: True when the bolt screws into the last member.

        Assign to switch an existing joint between a nut and a tapped
        hole; the bolt length is re-validated against the new rule.
        """
        return self._tapped

    @tapped.setter
    def tapped(self, value):
        value = bool(value)
        if self.plates is not None:
            self._validate_bolt_length(self.bolt, self.plates, value)
        self._tapped = value

    @property
    def grip(self):
        """float: Grip length (sum of member thicknesses) in mm.

        This is the physical stack. The stiffness chain uses
        :attr:`effective_grip`, which differs only for a tapped joint.

        Raises:
            ValueError: If no plates are defined.
        """
        self._require_plates()
        return sum(t for t, _ in self.plates)  # type: ignore

    @property
    def effective_grip(self):
        """float: Grip l' used by the stiffness chain, in mm.

        For a through-bolted joint this is :attr:`grip`. For a tapped
        joint the bolt is not clamped by a nut at the far face, so
        Shigley (Sec. 8-5, Fig. 8-21b) replaces the tapped member of
        thickness t2 by an effective half-thickness::

            l' = h + t2/2   if t2 < d
            l' = h + d/2    otherwise

        with h the total thickness of the members above the tapped one.

        Raises:
            ValueError: If no plates are defined.
        """
        self._require_plates()
        return sum(t for t, _ in self._effective_stack())

    def _effective_stack(self):
        """Member stack summing to :attr:`effective_grip`.

        Identity for a through-bolted joint; for a tapped joint the last
        member is truncated to its effective half-thickness.
        """
        plates = self.plates
        if not self._tapped:
            return list(plates)  # type: ignore
        thickness, material = plates[-1]  # type: ignore
        diameter = self.bolt.nominal_diameter
        effective = thickness / 2 if thickness < diameter else diameter / 2
        return list(plates[:-1]) + [(effective, material)]  # type: ignore

    @property
    def bolt_stiffness(self):
        """float: Bolt stiffness kb in N/mm (Shigley Eq. 8-17).

        The segmented model: the unthreaded shank (nominal area Ad over
        ld) and the threaded portion (stress area At over lt) act as
        springs in series over the grip, giving
        kb = Ad*At*E / (Ad*lt + At*ld). Uses :attr:`effective_grip` --
        the stretched length in the joint model -- not ``bolt.length``;
        see :meth:`mecapy.bolts.Bolt.segmented_stiffness`.

        Raises:
            ValueError: If no plates are defined.
        """
        return self.bolt.segmented_stiffness(self.effective_grip)

    def _member_frusta(self, diameter, alpha_deg=30.0):
        """Build the frustum stack for Shigley's member-stiffness model.

        The clamp zone is a pair of cones (half-angle ``alpha_deg``,
        30 degrees by default) spreading from each washer face
        (dw = 1.5*d) toward the grip midplane. Each plate segment
        between a face and the midplane is one frustum; a plate
        straddling the midplane is split there.

        The stack is :meth:`_effective_stack`, so a tapped joint is
        built over l' rather than the physical grip. Mirroring that
        stack for the second half is Shigley's tapped-joint assumption
        (Sec. 8-5): the joint is treated as symmetric about the
        midplane of l', since there is no nut washer face to spread
        from. It is deliberate, not an oversight.

        Args:
            diameter (float): Bolt nominal diameter in mm.
            alpha_deg (float): Cone half-angle in degrees (default: 30,
                Shigley's standard value).

        Returns:
            list: ``(thickness, small_diameter, elastic_modulus)`` per
            frustum, in mm / mm / MPa.
        """
        dw = WASHER_FACE_RATIO * diameter
        stack = self._effective_stack()
        mid = sum(t for t, _ in stack) / 2
        cone_factor = 2 * tan(radians(alpha_deg))
        frusta = []

        def add_half(layers):
            """Walk layers inward from one face up to the midplane."""
            z = 0.0
            for thickness, material in layers:
                if z >= mid:
                    break
                t = min(thickness, mid - z)
                small_d = dw + cone_factor * z
                modulus = get_material_properties(material)["elastic_modulus"] / 1e6
                frusta.append((t, small_d, modulus))
                z += thickness

        add_half(stack)
        add_half(list(reversed(stack)))
        return frusta

    def _member_stiffness_for(self, diameter, alpha_deg=30.0):
        """Member stiffness km in N/mm for a given bolt diameter.

        Args:
            diameter (float): Bolt nominal diameter in mm.
            alpha_deg (float): Cone half-angle in degrees (default: 30,
                Shigley's standard value); passed through to
                :meth:`_member_frusta` so the frustum geometry and the
                stiffness formula use the same angle.

        Raises:
            ValueError: If ``alpha_deg`` is not strictly between 0 and 90.
        """
        if not 0 < alpha_deg < 90:
            raise ValueError("Cone half-angle must be strictly between 0 and 90 degrees")
        tan_alpha = tan(radians(alpha_deg))
        total_compliance = 0.0
        for t, small_d, modulus in self._member_frusta(diameter, alpha_deg):
            d = diameter
            ln_arg = (
                (2 * tan_alpha * t + small_d - d) * (small_d + d)
            ) / (
                (2 * tan_alpha * t + small_d + d) * (small_d - d)
            )
            k = tan_alpha * pi * modulus * d / log(ln_arg)
            total_compliance += 1 / k
        return 1 / total_compliance

    @property
    def member_stiffness(self):
        """float: Member stiffness km in N/mm (Shigley frustum cones).

        Each plate contributes frusta (half-angle :attr:`cone_angle_deg`,
        30 degrees by default) growing from the washer faces
        (dw = 1.5*d) toward the grip midplane; the frusta act as springs
        in series.

        Raises:
            ValueError: If no plates are defined.
        """
        self._require_plates()
        return self._member_stiffness_for(self.bolt.nominal_diameter, self.cone_angle_deg)

    @property
    def joint_constant(self):
        """float: Joint constant C = kb / (kb + km).

        Fraction of an external tensile load carried by the bolt; the
        members carry (1 - C).

        Raises:
            ValueError: If no plates are defined.
        """
        kb = self.bolt_stiffness
        return kb / (kb + self.member_stiffness)

    def _joint_constant_for(self, bolt):
        """Joint constant C for a candidate bolt on the current plates.

        Uses the same segmented kb as :attr:`bolt_stiffness`, so the
        sizing walk in :meth:`minimum_bolt` and the reported
        :attr:`joint_constant` cannot disagree.
        """
        kb = bolt.segmented_stiffness(self.effective_grip)
        return kb / (kb + self._member_stiffness_for(bolt.nominal_diameter, self.cone_angle_deg))

    @property
    def effective_preload(self):
        """float: Per-bolt preload Fi in N.

        The explicit :attr:`preload` if set, otherwise the bolt's
        recommended preload (0.75 of the proof load).
        """
        if self.preload is not None:
            return self.preload
        return self.bolt.recommended_preload

    # ---- Preloaded joint analysis ----

    def bolt_tensions(self):
        """
        Distribute preload and external tension between bolts and members.

        With joint constant C and preload Fi, an external per-bolt load
        P puts Fb = Fi + C*P on the bolt and Fm = (1 - C)*P - Fi on the
        members (negative while the joint stays clamped).

        Returns:
            dict: ``{bolt_number: {"external": P, "bolt_tension": Fb,
            "member_force": Fm}}`` in N, with P the signed axial load
            from :meth:`bolt_forces`.

        Raises:
            ValueError: If no plates are defined.
        """
        self._require_plates()
        c = self.joint_constant
        fi = self.effective_preload
        result = {}
        for number, entry in self.bolt_forces().items():
            p = entry["axial"]
            result[number] = {
                "external": p,
                "bolt_tension": fi + c * p,
                "member_force": (1 - c) * p - fi,
            }
        return result

    def separation_safety_factors(self):
        """
        Safety factor against joint separation for each bolt.

        n0 = Fi / ((1 - C) * P) (Shigley eq. 8-30): the ratio of preload
        to the member decompression caused by the external tension P.

        Returns:
            dict: ``{bolt_number: n0}``. Bolts without external tension
            (P <= 0) get ``float("inf")``.

        Raises:
            ValueError: If no plates are defined.
        """
        self._require_plates()
        c = self.joint_constant
        fi = self.effective_preload
        result = {}
        for number, entry in self.bolt_forces().items():
            p = entry["axial"]
            if p > 0:
                result[number] = fi / ((1 - c) * p)
            else:
                result[number] = float("inf")
        return result

    def proof_safety_factors(self):
        """
        Load factor against exceeding the proof strength for each bolt.

        nL = (Sp*At - Fi) / (C * P) (Shigley eq. 8-29): the factor by
        which the external tension P can grow before the bolt stress
        reaches the proof strength.

        Returns:
            dict: ``{bolt_number: nL}``. Bolts without external tension
            (P <= 0) get ``float("inf")``.

        Raises:
            ValueError: If no plates are defined.
        """
        self._require_plates()
        c = self.joint_constant
        fi = self.effective_preload
        capacity = self.bolt.proof_strength * self.bolt.stress_area - fi
        result = {}
        for number, entry in self.bolt_forces().items():
            p = entry["axial"]
            if p > 0:
                result[number] = capacity / (c * p)
            else:
                result[number] = float("inf")
        return result

    def fatigue_safety_factor(self, external_load_max, external_load_min=0.0,
                              endurance_limit=None):
        """
        Bolt fatigue safety factor for a fluctuating external load (Ch. 8).

        For a per-bolt external tension cycling between ``external_load_min``
        and ``external_load_max``, only the joint-constant fraction C reaches
        the bolt. The alternating and mean bolt stresses are (Shigley
        Eq. 8-35, with sigma_i = Fi/At the preload stress)::

            sigma_a = C (P_max - P_min) / (2 At)
            sigma_m = sigma_i + C (P_max + P_min) / (2 At)

        and the Goodman fatigue factor along the load line from the preload
        point (Eq. 8-38) is::

            nf = Se (Sut - sigma_i) / (Sut sigma_a + Se (sigma_m - sigma_i))

        Args:
            external_load_max (float): Peak external tension per bolt in N
                (or a pint.Quantity force).
            external_load_min (float): Minimum external tension per bolt in
                N or a pint.Quantity (default 0, a released-to-zero
                repeated load).
            endurance_limit (float): Fully corrected bolt endurance limit
                Se in MPa (e.g. Shigley Table 8-17, which already includes
                the thread stress concentration). Defaults to the bolt
                material's Marin endurance limit — an approximation, since
                that omits the rolled-thread Kf; pass the tabulated Se for a
                real bolt.

        Returns:
            float: Fatigue safety factor; ``inf`` if there is no
            alternating load.

        Raises:
            ValueError: If no plates are defined or
                ``external_load_max < external_load_min``.
        """
        self._require_plates()
        external_load_max = to_magnitude(external_load_max, "N")
        external_load_min = to_magnitude(external_load_min, "N")
        if external_load_max < external_load_min:
            raise ValueError(
                "external_load_max must be >= external_load_min")
        at = self.bolt.stress_area
        c = self.joint_constant
        sigma_i = self.effective_preload / at
        sut = self.bolt.tensile_strength
        se = (endurance_limit if endurance_limit is not None
              else self.bolt._fatigue_material().endurance_limit)
        pa = (external_load_max - external_load_min) / 2.0
        pm = (external_load_max + external_load_min) / 2.0
        sigma_a = c * pa / at
        if sigma_a <= 0:
            return float("inf")
        sigma_m = sigma_i + c * pm / at
        return se * (sut - sigma_i) / (sut * sigma_a + se * (sigma_m - sigma_i))

    def slip_safety_factors(self, mu=DEFAULT_MU):
        """
        Safety factor against slip at the clamped interface per bolt.

        The clamp force left at the interface is Fi - (1 - C)*P (a
        compressive external load increases it); the friction capacity
        mu * clamp is compared with the bolt's resultant shear V.

        Args:
            mu (float): Interface friction coefficient (default:
                ``DEFAULT_MU``, 0.3 -- dry steel on steel).

        Returns:
            dict: ``{bolt_number: mu * max(Fi - (1-C)*P, 0) / V}``.
            Bolts without shear get ``float("inf")``.

        Raises:
            ValueError: If no plates are defined or ``mu`` is not
                strictly positive.
        """
        self._require_plates()
        if mu <= 0:
            raise ValueError("Friction coefficient must be strictly positive")
        c = self.joint_constant
        fi = self.effective_preload
        result = {}
        for number, entry in self.bolt_forces().items():
            clamp = fi - (1 - c) * entry["axial"]
            capacity = mu * max(clamp, 0.0)
            v = entry["shear_magnitude"]
            result[number] = capacity / v if v > 0 else float("inf")
        return result

    # ---- Member (plate) checks ----

    @staticmethod
    def _plate_yield(material):
        """Yield strength of a member material in MPa.

        Raises:
            ValueError: If the material has no tabulated yield strength.
        """
        properties = get_material_properties(material)
        try:
            return properties["yield_strength"] / 1e6
        except KeyError:
            raise ValueError(
                f"Material {material!r} has no yield strength in the "
                f"database; the member checks need one"
            )

    @property
    def hole_diameter(self):
        """float: Diameter of the holes through the members, in mm.

        Defaults to the bolt's nominal diameter (no clearance). It is
        used only where the missing material matters -- the washer-face
        annulus of :meth:`clamp_states`. Bearing and tear-out use the
        bolt diameter itself, since it is the shank that pushes on the
        member.
        """
        if self._hole_diameter is None:
            return self.bolt.nominal_diameter
        return self._hole_diameter

    @hole_diameter.setter
    def hole_diameter(self, value):
        if value is None:
            self._hole_diameter = None
            return
        value = to_magnitude(value, "mm")
        diameter = self.bolt.nominal_diameter
        if value < diameter:
            raise ValueError(
                f"Hole diameter {value} mm is smaller than the bolt "
                f"diameter {diameter} mm"
            )
        washer = WASHER_FACE_RATIO * diameter
        if value >= washer:
            raise ValueError(
                f"Hole diameter {value} mm leaves no washer face "
                f"(dw = {WASHER_FACE_RATIO}*d = {washer} mm)"
            )
        self._hole_diameter = float(value)

    def bearing_stresses(self):
        """
        Bearing (crushing) stress of each bolt on each member.

        The bolt shank pushes on the hole wall over its projected area,
        so with the bolt's resultant shear V and a member of thickness
        t (Shigley Sec. 8-11)::

            sigma = V / (d * t)

        The projected width is the bolt's nominal diameter, not the hole
        diameter: it is the shank that bears. This matches
        :meth:`mecapy.joints.Rivet.bearing_stress`.

        Returns:
            dict: ``{bolt_number: {"shear": V, "safety_factor": n,
            "members": [{"thickness", "material", "stress",
            "yield_strength", "safety_factor"}, ...]}}`` with stresses in
            MPa and ``n`` the governing (smallest) member factor.

        Raises:
            ValueError: If no plates are defined, or a member material
                has no yield strength.
        """
        self._require_plates()
        diameter = self.bolt.nominal_diameter
        result = {}
        for number, entry in self.bolt_forces().items():
            shear = entry["shear_magnitude"]
            members = []
            for thickness, material in self.plates:  # type: ignore
                strength = self._plate_yield(material)
                stress = shear / (diameter * thickness)
                members.append({
                    "thickness": thickness,
                    "material": material,
                    "stress": stress,
                    "yield_strength": strength,
                    "safety_factor": strength / stress if stress > 0 else float("inf"),
                })
            result[number] = {
                "shear": shear,
                "safety_factor": min(m["safety_factor"] for m in members),
                "members": members,
            }
        return result

    def bearing_safety_factors(self):
        """
        Governing bearing safety factor of the members, per bolt.

        Returns:
            dict: ``{bolt_number: n}``, the smallest member factor from
            :meth:`bearing_stresses`. Bolts without shear get
            ``float("inf")``.

        Raises:
            ValueError: If no plates are defined.
        """
        return {
            number: entry["safety_factor"]
            for number, entry in self.bearing_stresses().items()
        }

    def clamp_states(self):
        """
        Clamp remaining at each bolt, and the pressure under its head.

        Turns the ``member_force`` of :meth:`bolt_tensions` into a
        check. While the joint holds, Fm is negative (compression) and
        the clamp left at the interface is -Fm; once Fm reaches 0 the
        members have decompressed and the joint has separated.

        The washer face bears with the full bolt tension Fb (the head
        pushes on the member with everything the bolt carries, not with
        the residual interface clamp) over the annulus between the
        washer face dw = 1.5*d and :attr:`hole_diameter`.

        Returns:
            dict: ``{bolt_number: {"external", "bolt_tension",
            "member_force", "clamp", "separated", "washer_pressure",
            "washer_safety_factor"}}`` in N / MPa. ``clamp`` is floored
            at 0 and ``separated`` is True once Fm >= 0. The washer
            factor uses the bolt-head-side member, and for a
            through-bolted joint also the nut-side member, taking the
            smaller.

        Raises:
            ValueError: If no plates are defined, or a member material
                has no yield strength.
        """
        self._require_plates()
        diameter = self.bolt.nominal_diameter
        washer_face = WASHER_FACE_RATIO * diameter
        area = pi / 4 * (washer_face ** 2 - self.hole_diameter ** 2)
        faces = [self.plates[0][1]]  # type: ignore
        if not self._tapped:
            faces.append(self.plates[-1][1])  # type: ignore
        strength = min(self._plate_yield(material) for material in faces)
        result = {}
        for number, entry in self.bolt_tensions().items():
            member_force = entry["member_force"]
            pressure = entry["bolt_tension"] / area
            result[number] = {
                "external": entry["external"],
                "bolt_tension": entry["bolt_tension"],
                "member_force": member_force,
                "clamp": max(-member_force, 0.0),
                "separated": member_force >= 0,
                "washer_pressure": pressure,
                "washer_safety_factor": (
                    strength / pressure if pressure > 0 else float("inf")
                ),
            }
        return result

    def minimum_edge_distances(self, safety_factor=1.0):
        """
        Smallest edge distance that prevents shear tear-out, per bolt.

        The bolt can tear a slug out to the free edge along two shear
        planes of area t*(e - d/2) each. Setting the capacity
        2*t*(e - d/2)*0.577*Sy equal to ``safety_factor`` times the
        bolt's shear V and solving for e gives::

            e = d/2 + n*V / (2 * t * 0.577 * Sy)

        measured from the bolt centre (the datum the ``positions`` use),
        so it is directly comparable with a coordinate. This is an
        output, not a constraint: the union never requires an edge
        distance as an input.

        Args:
            safety_factor (float): Target factor n against tear-out
                (default: 1.0).

        Returns:
            dict: ``{bolt_number: {"shear": V, "edge_distance": e,
            "members": [{"thickness", "material", "edge_distance"}, ...]}}``
            in N / mm, with ``e`` the governing (largest) requirement
            over the members. A bolt without shear needs only d/2.

        Raises:
            ValueError: If no plates are defined, ``safety_factor`` is
                not strictly positive, or a member material has no yield
                strength.
        """
        self._require_plates()
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        diameter = self.bolt.nominal_diameter
        result = {}
        for number, entry in self.bolt_forces().items():
            shear = entry["shear_magnitude"]
            members = []
            for thickness, material in self.plates:  # type: ignore
                strength = self._plate_yield(material)
                capacity = 2 * thickness * SHEAR_YIELD_FACTOR * strength
                members.append({
                    "thickness": thickness,
                    "material": material,
                    "edge_distance": diameter / 2 + safety_factor * shear / capacity,
                })
            result[number] = {
                "shear": shear,
                "edge_distance": max(m["edge_distance"] for m in members),
                "members": members,
            }
        return result

    # ---- Sizing ----

    def _sizing_demands(self):
        """Per-bolt (shear, tension) demands with tension clamped to >= 0."""
        demands = [
            (entry["shear_magnitude"], max(entry["axial"], 0.0))
            for entry in self.bolt_forces().values()
        ]
        if all(v == 0 and p == 0 for v, p in demands):
            raise ValueError(
                "Union carries no shear or tensile load; nothing to size"
            )
        return demands

    def required_stress_area(self, mu=DEFAULT_MU, property_class="8.8",
                             safety_factor=1.0):
        """
        Minimum tensile stress area for a slip-critical joint.

        Assumes the reusable-joint preload rule Fi = 0.75 * Sp * At and
        that friction carries all shear. Per bolt, At must satisfy the
        slip, proof (load-factor) and separation checks of
        :meth:`minimum_bolt` with design factor ``safety_factor``; the
        result is the largest bound over all bolts.

        When plates are defined the joint constant is evaluated at the
        *current* bolt diameter (an approximation — C changes slightly
        with the candidate diameter; :meth:`minimum_bolt` re-evaluates
        it exactly). Without plates, conservative bounds are used:
        C = 1 for the bolt-tension check and C = 0 for the clamp checks.

        Args:
            mu (float): Interface friction coefficient (default:
                ``DEFAULT_MU``, 0.3 -- dry steel on steel).
            property_class (str): ISO 898-1 class for the proof strength
                (default: "8.8"). Pass None to fall back to the current
                bolt's class instead.
            safety_factor (float): Design factor applied to the checks
                (default: 1.0).

        Returns:
            float: Minimum tensile stress area in mm^2.

        Raises:
            ValueError: If ``mu`` or ``safety_factor`` is not strictly
                positive, the property class is unknown, or the union
                carries no load.
        """
        if mu <= 0:
            raise ValueError("Friction coefficient must be strictly positive")
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        pc = property_class if property_class is not None else self.bolt.property_class
        sp = get_property_class(pc)["proof_strength"]
        if self.plates is not None:
            c_bolt = c_clamp = self.joint_constant
        else:
            c_bolt, c_clamp = 1.0, 0.0
        n = safety_factor
        at_min = 0.0
        for v, p in self._sizing_demands():
            at_slip = (n * v / mu + (1 - c_clamp) * p) / (0.75 * sp)
            at_proof = n * c_bolt * p / (0.25 * sp)
            at_sep = n * (1 - c_clamp) * p / (0.75 * sp)
            at_min = max(at_min, at_slip, at_proof, at_sep)
        return at_min

    def minimum_bolt(self, mu=DEFAULT_MU, property_class="8.8", safety_factor=1.0):
        """
        Find the smallest table bolt that supports the union loads.

        Candidates are drawn from the same thread family as the current
        bolt: the ISO coarse table for a metric (or custom) bolt, and the
        matching Unified series (UNC or UNF) for an imperial one.
        Models a slip-critical joint: shear is carried entirely by
        friction at the clamped interface, with the reusable-joint
        preload Fi = 0.75 * Sp * At (``recommended_preload``). Each
        candidate size must pass three checks for every bolt, with
        design factor n = ``safety_factor``, V the bolt shear and P its
        external tension:

        - slip:        mu * (Fi - (1 - C)*P) >= n * V
        - proof:       Sp*At >= Fi + n * C * P   (load factor nL >= n)
        - separation:  Fi >= n * (1 - C)*P       (n0 >= n)

        When plates are defined, C is recomputed for each candidate
        diameter from the frustum model. Without plates, conservative
        bounds are used (C = 1 for the bolt-tension check, C = 0 for the
        clamp checks). The union is not modified; apply the result with
        :meth:`set_bolt`.

        Args:
            mu (float): Interface friction coefficient (default:
                ``DEFAULT_MU``, 0.3 -- dry steel on steel).
            property_class (str): ISO 898-1 class or SAE J429 grade of the
                candidate bolts (default: "8.8"). Pass None to fall back
                to the current bolt's class instead.
            safety_factor (float): Design factor for the three checks
                (default: 1.0).

        Returns:
            Bolt: Smallest passing bolt, with the current bolt's length
            and material.

        Raises:
            ValueError: If the inputs are invalid, the union carries no
                load, or no size in the candidate table passes.
        """
        if mu <= 0:
            raise ValueError("Friction coefficient must be strictly positive")
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        pc = property_class if property_class is not None else self.bolt.property_class
        get_property_class(pc)  # validate the class/grade up front
        demands = self._sizing_demands()
        n = safety_factor
        series = self.bolt.thread_series
        if series in ("UNC", "UNF"):
            table = {s: d for s, d in UNIFIED_THREADS.items() if d["series"] == series}
        else:
            table = ISO_COARSE_THREADS
        sizes = sorted(table, key=lambda s: table[s]["stress_area"])
        for size in sizes:
            candidate = Bolt(size, length=self.bolt.length, property_class=pc,
                             material=self.bolt.material)
            at = candidate.stress_area
            # Sp is size dependent for SAE grades 2 and 5, so resolve per
            # candidate; a grade may not cover the largest sizes at all.
            try:
                sp = candidate.proof_strength
            except ValueError:
                continue
            fi = candidate.recommended_preload
            if self.plates is not None:
                c_bolt = c_clamp = self._joint_constant_for(candidate)
            else:
                c_bolt, c_clamp = 1.0, 0.0
            ok = True
            for v, p in demands:
                slip = mu * (fi - (1 - c_clamp) * p) >= n * v
                proof = sp * at >= fi + n * c_bolt * p
                separation = fi >= n * (1 - c_clamp) * p
                if not (slip and proof and separation):
                    ok = False
                    break
            if ok:
                return candidate
        raise ValueError(
            f"No bolt up to {sizes[-1]} satisfies the load with mu={mu}, "
            f"property class {pc} and safety factor {safety_factor}"
        )

    # ---- Reporting ----

    def joint_report(self):
        """
        Every joint-level and per-bolt result in one dict.

        The machine-readable counterpart of :meth:`describe`. Without a
        member model the stiffness chain is unavailable, so those
        entries are None and the per-bolt rows carry only the force
        distribution rather than raising.

        Returns:
            dict: ``n_bolts``, ``grip``, ``effective_grip``, ``tapped``,
            ``bolt_stiffness``, ``member_stiffness``, ``joint_constant``,
            ``preload``, and ``bolts`` mapping each bolt number to
            ``shear``, ``axial``, and -- with plates -- ``bolt_tension``,
            ``member_force``, ``clamp``, ``separated``,
            ``separation``, ``proof``, ``bearing`` and ``edge_distance``.
        """
        forces = self.bolt_forces()
        report = {
            "n_bolts": self.n_bolts,
            "grip": None,
            "effective_grip": None,
            "tapped": self._tapped,
            "bolt_stiffness": None,
            "member_stiffness": None,
            "joint_constant": None,
            "preload": None,
            "bolts": {},
        }
        has_members = self.plates is not None
        if has_members:
            report.update({
                "grip": self.grip,
                "effective_grip": self.effective_grip,
                "bolt_stiffness": self.bolt_stiffness,
                "member_stiffness": self.member_stiffness,
                "joint_constant": self.joint_constant,
                "preload": self.effective_preload,
            })
            clamps = self.clamp_states()
            separation = self.separation_safety_factors()
            proof = self.proof_safety_factors()
            bearing = self.bearing_safety_factors()
            edges = self.minimum_edge_distances()
        for number, entry in forces.items():
            row = {
                "shear": entry["shear_magnitude"],
                "axial": entry["axial"],
            }
            if has_members:
                state = clamps[number]
                row.update({
                    "bolt_tension": state["bolt_tension"],
                    "member_force": state["member_force"],
                    "clamp": state["clamp"],
                    "separated": state["separated"],
                    "washer_safety_factor": state["washer_safety_factor"],
                    "separation": separation[number],
                    "proof": proof[number],
                    "bearing": bearing[number],
                    "edge_distance": edges[number]["edge_distance"],
                })
            report["bolts"][number] = row
        return report

    def describe(self):
        """
        Multi-line summary of the joint and every bolt in it.

        Joint-level lines are formatted as ``label (symbol) = value
        unit`` like :meth:`mecapy.gears.Gear.describe`, followed by a
        per-bolt table. Without plates the member model is reported as
        absent and only the force distribution is tabulated, so this
        stays usable on a bare bolt group. The string is returned, not
        printed; use ``print(union.describe())``.

        Returns:
            str: Formatted joint report.
        """
        report = self.joint_report()
        header = f"{self.__class__.__name__} joint"
        if self.name:
            header += f" '{self.name}'"
        lines = [header, "=" * 72, f"bolts (n) = {report['n_bolts']}"]
        lines.append(f"bolt = {self.bolt.size} class {self.bolt.property_class}")
        if report["joint_constant"] is None:
            lines.append("member model: none (no plates defined)")
            lines.append("")
            lines.append(f"{'#':>3} {'|Fs| [N]':>12} {'P [N]':>12}")
            lines.append("-" * 72)
            for number, row in report["bolts"].items():
                lines.append(
                    f"{number:>3} {row['shear']:>12.1f} {row['axial']:>12.1f}"
                )
            return "\n".join(lines)

        lines.extend([
            f"grip (l) = {report['grip']:.2f} mm",
            f"effective grip (l') = {report['effective_grip']:.2f} mm"
            + (" (tapped)" if report["tapped"] else ""),
            f"bolt stiffness (kb) = {report['bolt_stiffness']:.0f} N/mm",
            f"member stiffness (km) = {report['member_stiffness']:.0f} N/mm",
            f"joint constant (C) = {report['joint_constant']:.4f}",
            f"preload (Fi) = {report['preload'] / 1000:.2f} kN",
            "",
            f"{'#':>3} {'|Fs|':>9} {'P':>9} {'Fb':>10} {'Fm':>10} "
            f"{'n_sep':>7} {'n_proof':>8} {'n_bear':>7} {'e_min':>7}",
            "-" * 72,
        ])
        for number, row in report["bolts"].items():
            lines.append(
                f"{number:>3} {row['shear']:>9.1f} {row['axial']:>9.1f} "
                f"{row['bolt_tension']:>10.1f} {row['member_force']:>10.1f} "
                f"{row['separation']:>7.2f} {row['proof']:>8.2f} "
                f"{row['bearing']:>7.2f} {row['edge_distance']:>7.2f}"
            )
        lines.append("(forces in N, e_min in mm from the bolt centre)")
        return "\n".join(lines)

    # ---- Visualization ----

    def plot_distribution(self, scale=None, show=True, ax=None, labels=True):
        """
        Plot how the applied loads distribute over the bolts.

        Each bolt is drawn at its position with the direct shear split
        into its axis-aligned components Vx and Vy (from Fx, Fy) in blue,
        and the torsion shear T (from Mz) as a single arrow along its
        true tangential direction in red -- torsion has no meaningful
        axis split, it acts perpendicular to the radius. Their vector sum
        R is the thin dashed gray arrow, drawn only when more than one
        contribution is present (with just one, R would sit exactly on
        top of it). Each bolt carries a text block with its number, the
        resultant shear R and the signed axial load, and the group
        centroid is marked with a cross. A bolt the joint bends about
        (see :attr:`bending_pivots`) is tagged "pivot Mx" / "pivot My"
        in its text block.

        Args:
            scale (float): Arrow scale in mm per N, applied linearly. If
                None (default), the largest shear magnitude is scaled to
                about 20% of the plot span and smaller ones follow a
                square-root compression instead of a straight linear one,
                so they stay visible instead of shrinking toward zero
                next to a much larger force. The compression is applied
                to each arrow's magnitude and its true direction is then
                reapplied, so no arrow's angle is distorted.
            show (bool): Call ``plt.show()`` (default: True). Pass False
                when embedding or testing.
            ax (matplotlib.axes.Axes): Axes to draw on. If None, a new
                figure is created.
            labels (bool): Annotate each arrow with its magnitude
                (default: True). Pass False for a dense pattern, where
                neighbouring bolts' arrow labels start to collide; the
                arrows and the per-bolt text blocks are still drawn.

        Returns:
            matplotlib.figure.Figure: The figure containing the plot.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_distribution; "
                "install it with 'pip install matplotlib'"
            )

        forces = self.bolt_forces()
        pivots = self.bending_pivots
        xs = [row[1] for row in self.positions]
        ys = [row[2] for row in self.positions]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = ax.figure

        span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
        # Scale against the largest arrow actually drawn: the direct
        # shear appears as two axis components, torsion and the resultant
        # as whole vectors.
        max_component = max(
            (
                value
                for entry in forces.values()
                for value in (
                    abs(entry["shear_direct"][0]),
                    abs(entry["shear_direct"][1]),
                    hypot(*entry["shear_torsion"]),
                    hypot(*entry["shear"]),
                )
            ),
            default=0.0,
        )
        target_length = 0.2 * span

        def arrow_offset(value):
            """Signed on-plot length (mm) for a force component/magnitude.

            With the auto scale (``scale`` argument left as None), uses a
            square-root compression of the magnitude instead of a straight
            linear one, so a small component next to a much larger one is
            still visibly drawn instead of shrinking toward zero; the
            largest component still maps to ``target_length`` either way.
            An explicit ``scale`` (mm/N) bypasses this and stays linear,
            since the caller asked for that specific rate.
            """
            if value == 0 or max_component == 0:
                return 0.0
            if scale is not None:
                return value * scale
            return copysign(sqrt(abs(value) / max_component) * target_length, value)

        marker_size = 100
        ax.scatter(xs, ys, s=marker_size, color="#374151", zorder=3, label="Bolts")
        x_bar, y_bar = self.centroid
        ax.plot(x_bar, y_bar, "+", color="#6b7280", markersize=12,
                markeredgewidth=2, zorder=2, label="Centroid")

        # Every point that must stay inside the view. Arrow patches are
        # clipped to the axes, so sizing the view from the bolts alone
        # silently amputates arrows that reach past the outermost bolt --
        # collect the tips as they are drawn and fit the limits at the end.
        extent = [(x, y) for _, x, y in self.positions]

        direct_color = "#3c25eb"
        torsion_color = "#d90b0b"
        resultant_color = "#9ca3af"
        offset = 0.03 * span

        def draw_vector(x, y, fx, fy, label, color, label_side,
                        linewidth=2, linestyle="-", mutation_scale=14,
                        zorder=4):
            """One arrow for a shear vector, drawn along its true direction.

            The square-root compression in ``arrow_offset`` is applied to
            the magnitude and the unit direction reapplied afterwards --
            compressing fx and fy separately would rotate the arrow.
            ``label_side`` (+1/-1) pushes the text sideways off the shaft
            so labels stay apart when two vectors are collinear. A
            ``label`` of None -- or ``labels=False`` on the call -- draws
            the arrow with no floating text.
            """
            magnitude = hypot(fx, fy)
            if magnitude == 0:
                return
            ux, uy = fx / magnitude, fy / magnitude
            length = arrow_offset(magnitude)
            tip = (x + ux * length, y + uy * length)
            extent.append(tip)
            ax.annotate(
                "", xy=tip, xytext=(x, y),
                arrowprops={"arrowstyle": "-|>", "color": color,
                            "linewidth": linewidth, "linestyle": linestyle,
                            "mutation_scale": mutation_scale},
                zorder=zorder,
            )
            if label is None or not labels:
                return
            # Text sits just past the tip, nudged along the perpendicular
            # (-uy, ux) so a collinear V and T do not overprint.
            text_x = tip[0] + ux * 0.5 * offset - uy * label_side * 1.1 * offset
            text_y = tip[1] + uy * 0.5 * offset + ux * label_side * 1.1 * offset
            extent.append((text_x, text_y))
            ax.annotate(
                f"{label} = {magnitude:.0f} N",
                xy=tip, xytext=(text_x, text_y),
                fontsize=8, color=color, zorder=5,
                ha="left" if ux >= 0 else "right", va="center",
            )

        for row in self.positions:
            number, x, y = row
            entry = forces[number]
            vx, vy = entry["shear_direct"]
            tx, ty = entry["shear_torsion"]
            fsx, fsy = entry["shear"]
            # Direct shear on the axes (it is just Fx/n and Fy/n, so the
            # split is meaningful); torsion as one tangential vector.
            draw_vector(x, y, vx, 0, "Vx", direct_color, +1)
            draw_vector(x, y, 0, vy, "Vy", direct_color, +1)
            draw_vector(x, y, tx, ty, "T", torsion_color, -1)
            # Only draw the resultant when it combines more than one
            # contribution: with just one, it is identical (same direction
            # and magnitude) to the arrow already drawn, so overlaying it
            # just doubles the arrowhead.
            contributions = sum(
                1 for c in (vx, vy, hypot(tx, ty)) if c != 0
            )
            if contributions > 1:
                # No floating label: a small V leaves R almost on top of
                # T, so a third label there just collides. Its magnitude
                # goes in the bolt's text block instead.
                draw_vector(x, y, fsx, fsy, None, resultant_color, 0,
                            linewidth=1, linestyle="--", mutation_scale=10,
                            zorder=3)
            # Put the bolt's text block in whichever of eight compass
            # directions is furthest from every arrow drawn at this bolt,
            # so it never lands under an arrow or its label. Picking the
            # direction that minimises the largest dot product maximises
            # the angular clearance to the closest arrow.
            arrow_dirs = [
                (cx / mag, cy / mag)
                for cx, cy in ((vx, 0), (0, vy), (tx, ty), (fsx, fsy))
                for mag in (hypot(cx, cy),)
                if mag > 0
            ]
            dx, dy = -0.7071, -0.7071
            if arrow_dirs:
                dx, dy = min(
                    (
                        (cos(k * pi / 4), sin(k * pi / 4))
                        for k in range(8)
                    ),
                    key=lambda c: max(c[0] * a[0] + c[1] * a[1]
                                      for a in arrow_dirs),
                )
            text_at = (x + dx * 2.0 * offset, y + dy * 2.0 * offset)
            extent.append(text_at)
            # Name the bending pivot(s): otherwise a bolt showing no
            # bending tension at all looks like a bug rather than the
            # line the joint rotates about.
            pivot_of = [axis for axis, pivot in pivots.items() if pivot == number]
            pivot_tag = (f" (pivot {'/'.join('M' + a for a in pivot_of)})"
                         if pivot_of else "")
            ax.annotate(
                f"#{number}{pivot_tag}\nR = {entry['shear_magnitude']:.0f} N\n"
                f"N = {entry['axial']:+.0f} N",
                xy=(x, y), xytext=text_at,
                fontsize=9, color="#374151", zorder=5,
                ha="left" if dx > 0.1 else ("right" if dx < -0.1 else "center"),
                va="bottom" if dy > 0.1 else ("top" if dy < -0.1 else "center"),
            )

        # Fit the view to everything drawn, with room for the label text
        # that hangs off each anchor point, then grow (never shrink) the
        # narrower of the two ranges to the wider one, so both axes are
        # already at the same scale. Equalizing here rather than leaving
        # it to adjustable="datalim" keeps the limits computed above:
        # that mode overrides pinned limits and logs "Ignoring fixed x
        # limits to fulfill fixed data aspect" every draw.
        ex = [p[0] for p in extent]
        ey = [p[1] for p in extent]
        pad = 0.22 * max(max(ex) - min(ex), max(ey) - min(ey), span)
        x0, x1 = min(ex) - pad, max(ex) + pad
        y0, y1 = min(ey) - pad, max(ey) + pad
        half = max(x1 - x0, y1 - y0) / 2
        x_mid, y_mid = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_aspect("equal", adjustable="box")

        ax.plot([], [], color=direct_color, linewidth=2,
                label="Direct shear (Vx, Vy)")
        ax.plot([], [], color=torsion_color, linewidth=2,
                label="Torsion shear (T)")
        ax.plot([], [], color=resultant_color, linewidth=1, linestyle="--",
                label="Resultant shear (R)")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Bolted union: force distribution")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        # Legend below the axes, never on top of them: the arrows and
        # their labels are annotations, which loc="best" does not see
        # when it looks for a clear spot, so an inset legend silently
        # covers whole bolts' worth of arrows.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
                  fontsize=9, frameon=False)

        if show:
            plt.show()
        return fig

    def plot_tension(self, show=True, ax=None, labels=True):
        """
        Plot how preload and external tension load each bolt.

        A companion to :meth:`plot_distribution`, which draws the
        in-plane shear. The axial load has no in-plane direction, so it
        gets its own bar chart rather than more arrows on the plan view.
        Each bolt shows a stacked bar of the preload Fi and the bolt's
        share C*P of the external load (together the resultant bolt load
        Fb) against a dashed proof-load line, and below the axis the
        clamp -Fm left on the members. A bar reaching the zero line from
        below means that joint has separated.

        Args:
            show (bool): Call ``plt.show()`` (default: True). Pass False
                when embedding or testing.
            ax (matplotlib.axes.Axes): Axes to draw on. If None, a new
                figure is created.
            labels (bool): Annotate each bolt with its Fb (default:
                True). Pass False for a dense pattern.

        Returns:
            matplotlib.figure.Figure: The figure containing the plot.

        Raises:
            ImportError: If matplotlib is not installed.
            ValueError: If no plates are defined.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_tension; "
                "install it with 'pip install matplotlib'"
            )

        self._require_plates()
        states = self.clamp_states()
        numbers = [row[0] for row in self.positions]
        fi = self.effective_preload
        preloads = [fi] * len(numbers)
        shares = [states[n]["bolt_tension"] - fi for n in numbers]
        clamps = [-states[n]["clamp"] for n in numbers]

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.figure

        spots = range(len(numbers))
        preload_color, share_color, clamp_color = "#3c25eb", "#d90b0b", "#0e9f6e"
        ax.bar(spots, preloads, color=preload_color, label="Preload (Fi)")
        ax.bar(spots, shares, bottom=preloads, color=share_color,
               label="Bolt share of the external load (C*P)")
        ax.bar(spots, clamps, color=clamp_color, label="Clamp on the members (-Fm)")

        proof = self.bolt.proof_load
        ax.axhline(proof, color="#374151", linestyle="--", linewidth=1,
                   label=f"Proof load ({proof / 1000:.1f} kN)")
        ax.axhline(0.0, color="#374151", linewidth=0.8)

        if labels:
            for spot, number in zip(spots, numbers):
                tension = states[number]["bolt_tension"]
                ax.annotate(f"{tension / 1000:.1f} kN", (spot, tension),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=8)

        ax.set_xticks(list(spots))
        ax.set_xticklabels([f"#{n}" for n in numbers])
        ax.set_xlabel("Bolt")
        ax.set_ylabel("Force [N]")
        ax.set_title("Bolted union: bolt and member load")
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
                  fontsize=9, frameon=False)

        if show:
            plt.show()
        return fig

    def __repr__(self):
        return (
            f"BoltedUnion(bolt={self.bolt!r}, n_bolts={self.n_bolts}, "
            f"forces={self.forces}, moments={self.moments})"
        )
