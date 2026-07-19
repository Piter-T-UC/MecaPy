"""Bolted union (bolt group) analysis module.

Distributes external loads over a pattern of identical bolts using the
elastic (rigid-plate) method. Units: coordinates in mm, forces in N,
moments in N*mm, stresses in MPa.

Load convention (right-handed axes, z out of the joint plane):
    - forces = (Fx, Fy, Fz): Fx, Fy are in-plane shear on the group,
      Fz is the axial (out-of-plane) force.
    - moments = (Mx, My, Mz): Mx and My are bending moments about the
      x and y axes, Mz is the torsion moment about z.
    - All loads act at the centroid of the bolt group.
"""

from math import copysign, log, pi, sqrt

from ..base import MechaElement
from ..materials import get_material_properties
from .bolt import Bolt
from .thread_data import ISO_COARSE_THREADS, get_property_class

# Default interface friction coefficient (dry steel on steel), matching
# the convention in mecapy.shafts.power_screw.
DEFAULT_MU = 0.3


class BoltedUnion(MechaElement):
    """
    Bolted union (bolt group) under combined loading.

    All bolts are identical (same :class:`Bolt` instance), so the
    elastic distribution depends only on the bolt positions: direct
    shear and axial force split equally, torsion produces shear
    proportional to the distance from the centroid, and bending
    produces axial force proportional to the distance from the
    centroidal axis.

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
        preload (float): Explicit per-bolt preload Fi in N, or None to
            use ``bolt.recommended_preload``.
    """

    def __init__(self, bolt, positions, forces=(0, 0, 0), moments=(0, 0, 0),
                 plates=None, preload=None, name=None):
        """
        Initialize a bolted union.

        Args:
            bolt (Bolt): Bolt instance used at every position.
            positions (list): List of ``[bolt_number, x, y]`` rows,
                coordinates in mm.
            forces (tuple): Applied forces (Fx, Fy, Fz) in N, acting at
                the bolt-group centroid (default: no force).
            moments (tuple): Applied moments (Mx, My, Mz) in N*mm about
                the centroid: Mx, My bending, Mz torsion (default: no
                moment).
            plates (list): Optional clamped stack as
                ``[(thickness, material), ...]`` (mm, material name)
                from bolt-head side to nut side. At least two plates.
            preload (float): Optional per-bolt preload Fi in N. Default
                None uses ``bolt.recommended_preload`` where needed.
            name (str): Optional identifier for the union.

        Raises:
            ValueError: If ``bolt`` is not a Bolt, a position row does
                not have exactly 3 entries, bolt numbers repeat, the
                force/moment vectors do not have 3 components, or the
                plates/preload are invalid.
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
        self.positions = [[row[0], float(row[1]), float(row[2])] for row in positions]
        self.forces = tuple(float(f) for f in forces)
        self.moments = tuple(float(m) for m in moments)
        self.plates = None
        self.preload = None
        self._centroid_override = None
        self.set_plates(plates)
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
            grip = sum(t for t, _ in self.plates)
            if bolt.length < grip:
                raise ValueError(
                    f"Bolt length {bolt.length} mm is shorter than the "
                    f"grip {grip} mm defined by the plates"
                )
        if self.preload is not None and self.preload > bolt.proof_load:
            raise ValueError(
                f"Preload {self.preload} N exceeds the proof load "
                f"{bolt.proof_load:.0f} N of the new bolt"
            )
        self.bolt = bolt
        self.material = bolt.material

    def set_plates(self, plates):
        """
        Define (or clear) the clamped plate stack.

        Args:
            plates (list): ``[(thickness, material), ...]`` from
                bolt-head side to nut side, thickness in mm and material
                a name from the material database. Pass None to remove
                the member model.

        Raises:
            ValueError: If a row is malformed, a thickness is not
                strictly positive, a material is unknown, fewer than two
                plates are given, or the bolt is shorter than the grip.
        """
        if plates is None:
            self.plates = None
            return
        normalized = []
        for row in plates:
            if len(row) != 2:
                raise ValueError(
                    f"Each plate must be (thickness, material); got {list(row)!r}"
                )
            thickness, material = row
            if thickness <= 0:
                raise ValueError("Plate thickness must be strictly positive")
            get_material_properties(material)  # raises for unknown materials
            normalized.append((float(thickness), material))
        if len(normalized) < 2:
            raise ValueError(
                "A joint clamps at least two plates (tapped-hole joints "
                "are not supported)"
            )
        grip = sum(t for t, _ in normalized)
        if self.bolt.length < grip:
            raise ValueError(
                f"Bolt length {self.bolt.length} mm is shorter than the "
                f"grip {grip} mm defined by the plates"
            )
        self.plates = normalized

    def set_preload(self, preload):
        """
        Set (or clear) the explicit per-bolt preload.

        Args:
            preload (float): Preload Fi in N, or None to fall back to
                ``bolt.recommended_preload``.

        Raises:
            ValueError: If ``preload`` is not strictly positive or
                exceeds the bolt's proof load.
        """
        if preload is None:
            self.preload = None
            return
        if preload <= 0:
            raise ValueError("Preload must be strictly positive")
        if preload > self.bolt.proof_load:
            raise ValueError(
                f"Preload {preload} N exceeds the bolt proof load "
                f"{self.bolt.proof_load:.0f} N"
            )
        self.preload = float(preload)

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
            point (tuple): (x, y) in mm, or None to remove the override
                and go back to the computed centroid.

        Raises:
            ValueError: If ``point`` is not None and does not have
                exactly 2 numeric components.
        """
        if point is None:
            self._centroid_override = None
            return
        if len(point) != 2:
            raise ValueError(f"centroid must be (x, y); got {list(point)!r}")
        self._centroid_override = (float(point[0]), float(point[1]))

    def _relative_coords(self):
        """Return [(number, dx, dy)] with coordinates relative to the centroid."""
        x_bar, y_bar = self.centroid
        return [(row[0], row[1] - x_bar, row[2] - y_bar) for row in self.positions]

    # ---- Load distribution ----

    def bolt_forces(self):
        """
        Distribute the applied loads over the bolts (elastic method).

        Per bolt:
            - direct shear: (Fx/n, Fy/n);
            - torsion Mz: shear of magnitude Mz*r_i / sum(r^2),
              perpendicular to the radius from the centroid;
            - direct axial: Fz/n;
            - bending Mx: axial Mx*dy_i / sum(dy^2) (right-hand rule:
              positive Mx puts bolts at positive y in tension);
            - bending My: axial -My*dx_i / sum(dx^2) (positive My puts
              bolts at negative x in tension).

        Returns:
            dict: ``{bolt_number: {"shear": (Fsx, Fsy),
            "shear_magnitude": float, "axial": float,
            "shear_direct": (Vx, Vy), "shear_torsion": (Tx, Ty)}}``
            with forces in N. ``shear`` is the component-wise sum of
            ``shear_direct`` (from Fx, Fy) and ``shear_torsion``
            (from Mz).

        Raises:
            ValueError: If a torsion/bending moment is non-zero but the
                bolt pattern has no lever arm to resist it (all bolts at
                the centroid or on the bending axis).
        """
        fx, fy, fz = self.forces
        mx, my, mz = self.moments
        n = self.n_bolts
        rel = self._relative_coords()

        sum_r2 = sum(dx ** 2 + dy ** 2 for _, dx, dy in rel)
        sum_dx2 = sum(dx ** 2 for _, dx, _dy in rel)
        sum_dy2 = sum(dy ** 2 for _, _dx, dy in rel)
        if mz != 0 and sum_r2 == 0:
            raise ValueError("Torsion applied but all bolts are at the centroid")
        if mx != 0 and sum_dy2 == 0:
            raise ValueError("Bending Mx applied but all bolts lie on the x axis")
        if my != 0 and sum_dx2 == 0:
            raise ValueError("Bending My applied but all bolts lie on the y axis")

        result = {}
        for number, dx, dy in rel:
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
            axial = fz / n
            if mx != 0:
                axial += mx * dy / sum_dy2
            if my != 0:
                axial += -my * dx / sum_dx2
            result[number] = {
                "shear": (fsx, fsy),
                "shear_magnitude": sqrt(fsx ** 2 + fsy ** 2),
                "axial": axial,
                "shear_direct": (vx, vy),
                "shear_torsion": (tx, ty),
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
    def grip(self):
        """float: Grip length (sum of plate thicknesses) in mm.

        Raises:
            ValueError: If no plates are defined.
        """
        self._require_plates()
        return sum(t for t, _ in self.plates)

    @property
    def bolt_stiffness(self):
        """float: Bolt stiffness kb = As * E / grip in N/mm.

        Uses the grip defined by the plates (the stretched length in the
        joint model), not ``bolt.length``.

        Raises:
            ValueError: If no plates are defined.
        """
        return self.bolt.stress_area * self.bolt.elastic_modulus / self.grip

    def _member_frusta(self, diameter):
        """Build the frustum stack for Shigley's member-stiffness model.

        The clamp zone is a pair of 30-degree cones spreading from each
        washer face (dw = 1.5*d) toward the grip midplane. Each plate
        segment between a face and the midplane is one frustum; a plate
        straddling the midplane is split there.

        Args:
            diameter (float): Bolt nominal diameter in mm.

        Returns:
            list: ``(thickness, small_diameter, elastic_modulus)`` per
            frustum, in mm / mm / MPa.
        """
        dw = 1.5 * diameter
        grip = self.grip
        mid = grip / 2
        frusta = []

        def add_half(layers):
            """Walk layers inward from one face up to the midplane."""
            z = 0.0
            for thickness, material in layers:
                if z >= mid:
                    break
                t = min(thickness, mid - z)
                small_d = dw + 1.155 * z  # cone at 30 degrees (2*tan30)
                modulus = get_material_properties(material)["elastic_modulus"] / 1e6
                frusta.append((t, small_d, modulus))
                z += thickness

        add_half(self.plates)
        add_half(list(reversed(self.plates)))
        return frusta

    def _member_stiffness_for(self, diameter):
        """Member stiffness km in N/mm for a given bolt diameter."""
        total_compliance = 0.0
        for t, small_d, modulus in self._member_frusta(diameter):
            d = diameter
            ln_arg = (
                (1.155 * t + small_d - d) * (small_d + d)
            ) / (
                (1.155 * t + small_d + d) * (small_d - d)
            )
            k = 0.5774 * pi * modulus * d / log(ln_arg)
            total_compliance += 1 / k
        return 1 / total_compliance

    @property
    def member_stiffness(self):
        """float: Member stiffness km in N/mm (Shigley frustum cones).

        Each plate contributes 30-degree frusta growing from the washer
        faces (dw = 1.5*d) toward the grip midplane; the frusta act as
        springs in series.

        Raises:
            ValueError: If no plates are defined.
        """
        self._require_plates()
        return self._member_stiffness_for(self.bolt.nominal_diameter)

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
        """Joint constant C for a candidate bolt on the current plates."""
        kb = bolt.stress_area * bolt.elastic_modulus / self.grip
        return kb / (kb + self._member_stiffness_for(bolt.nominal_diameter))

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

    def slip_safety_factors(self, mu=DEFAULT_MU):
        """
        Safety factor against slip at the clamped interface per bolt.

        The clamp force left at the interface is Fi - (1 - C)*P (a
        compressive external load increases it); the friction capacity
        mu * clamp is compared with the bolt's resultant shear V.

        Args:
            mu (float): Interface friction coefficient (default: 0.15).

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
            mu (float): Interface friction coefficient (default: 0.15).
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
        Find the smallest ISO coarse bolt that supports the union loads.
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
            mu (float): Interface friction coefficient (default: 0.15).
            property_class (str): ISO 898-1 class of the candidate bolts
                (default: "8.8"). Pass None to fall back to the current
                bolt's class instead.
            safety_factor (float): Design factor for the three checks
                (default: 1.0).

        Returns:
            Bolt: Smallest passing bolt, with the current bolt's length
            and material.

        Raises:
            ValueError: If the inputs are invalid, the union carries no
                load, or no ISO coarse size up to M36 passes.
        """
        if mu <= 0:
            raise ValueError("Friction coefficient must be strictly positive")
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        pc = property_class if property_class is not None else self.bolt.property_class
        sp = get_property_class(pc)["proof_strength"]
        demands = self._sizing_demands()
        n = safety_factor
        sizes = sorted(ISO_COARSE_THREADS, key=lambda s: ISO_COARSE_THREADS[s]["stress_area"])
        for size in sizes:
            candidate = Bolt(size, length=self.bolt.length, property_class=pc,
                             material=self.bolt.material)
            fi = candidate.recommended_preload
            at = candidate.stress_area
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
            f"No ISO coarse bolt up to M36 satisfies the load with mu={mu}, "
            f"property class {pc} and safety factor {safety_factor}"
        )

    # ---- Visualization ----

    def plot_distribution(self, scale=None, show=True, ax=None):
        """
        Plot how the applied loads distribute over the bolts.

        Each bolt is drawn at its position with the shear broken into
        axis-aligned components, each annotated with its signed
        magnitude: direct shear (Vx, Vy from Fx, Fy) in blue, torsion
        shear (Tx, Ty from Mz) in orange, and the resultant as a thin
        dashed gray arrow. The axial load is annotated as a signed
        magnitude next to the bolt. The group centroid is marked with a
        cross.

        Args:
            scale (float): Arrow scale in mm per N, applied linearly. If
                None (default), the largest shear component/magnitude is
                scaled to about 20% of the plot span and smaller ones
                follow a square-root compression instead of a straight
                linear one, so they stay visible instead of shrinking
                toward zero next to a much larger force.
            show (bool): Call ``plt.show()`` (default: True). Pass False
                when embedding or testing.
            ax (matplotlib.axes.Axes): Axes to draw on. If None, a new
                figure is created.

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
        xs = [row[1] for row in self.positions]
        ys = [row[2] for row in self.positions]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = ax.figure

        span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
        max_component = max(
            (abs(c) for entry in forces.values()
             for pair in (entry["shear_direct"], entry["shear_torsion"])
             for c in pair),
            default=0.0,
        )
        max_component = max(
            max_component,
            max((entry["shear_magnitude"] for entry in forces.values()), default=0.0),
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

        # Freeze the view now (aspect + margins) and render once so the
        # data<->pixel transform below is accurate; autoscale is then
        # switched off so the arrows/labels added later can't shift it.
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.25)
        fig.canvas.draw()
        ax.autoscale(False)

        # Arrow origins are not nudged away from the bolt marker (disabled).
        origin_shift = 0

        direct_color = "#3c25eb"
        torsion_color = "#d90b0b"
        offset = 0.03 * span

        def draw_component(x, y, cx, cy, label, value, color,
                           base_shift, label_shift, ha):
            """Arrow for one axis-aligned shear component with its signed value.

            base_shift nudges the arrow origin and label_shift the text so
            direct and torsion components along the same axis stay legible.
            """
            if value == 0:
                return
            base = (x + base_shift[0], y + base_shift[1])
            tip = (base[0] + arrow_offset(cx), base[1] + arrow_offset(cy))
            ax.annotate(
                "", xy=tip, xytext=base,
                arrowprops={"arrowstyle": "-|>", "color": color,
                            "linewidth": 2, "mutation_scale": 14},
                zorder=4,
            )
            ax.annotate(
                f"{label} = {value:+.0f} N",
                xy=tip, xytext=(tip[0] + label_shift[0], tip[1] + label_shift[1]),
                fontsize=8, color=color, zorder=5, ha=ha, va="center",
            )

        for row in self.positions:
            number, x, y = row
            entry = forces[number]
            vx, vy = entry["shear_direct"]
            tx, ty = entry["shear_torsion"]
            # Direct shear on the axes; torsion nudged off-axis so both
            # arrows and labels stay visible when they share a direction.
            # Horizontal labels go past the arrow tip on the side it
            # points to, so they never run back over the bolt text.
            vx_side = 1 if vx >= 0 else -1
            tx_side = 1 if tx >= 0 else -1
            draw_component(x, y, vx, 0, "Vx", vx, direct_color,
                           (0, origin_shift),
                           (vx_side * 0.3 * offset, 0.5 * offset),
                           "left" if vx_side > 0 else "right")
            draw_component(x, y, 0, vy, "Vy", vy, direct_color,
                           (origin_shift, 0), (0.5 * offset, 0.3 * offset), "left")
            draw_component(x, y, tx, 0, "Tx", tx, torsion_color,
                           (0, -origin_shift),
                           (tx_side * 0.3 * offset, -0.5 * offset),
                           "left" if tx_side > 0 else "right")
            draw_component(x, y, 0, ty, "Ty", ty, torsion_color,
                           (-origin_shift, 0), (-0.5 * offset, 0.3 * offset), "right")
            fsx, fsy = entry["shear"]
            nonzero_components = sum(1 for c in (vx, vy, tx, ty) if c != 0)
            # Only draw the resultant when it combines more than one
            # component: with a single active component it is identical
            # (same direction and magnitude) to the arrow already drawn
            # for it, so overlaying it just doubles the arrowhead.
            shear_mag = entry["shear_magnitude"]
            if shear_mag > 0 and nonzero_components > 1:
                # Scale by magnitude and reapply the true unit direction,
                # rather than offsetting fsx/fsy separately, since the
                # square-root compression is not linear (it would distort
                # the resultant's angle if applied per axis).
                length = arrow_offset(shear_mag)
                ux, uy = fsx / shear_mag, fsy / shear_mag
                ax.annotate(
                    "", xy=(x + ux * length, y + uy * length), xytext=(x, y),
                    arrowprops={"arrowstyle": "-|>", "color": "#9ca3af",
                                "linewidth": 1, "linestyle": "--",
                                "mutation_scale": 10},
                    zorder=3,
                )
            ax.annotate(
                f"#{number}\nN = {entry['axial']:+.0f} N",
                xy=(x, y), xytext=(x - 2.2 * offset, y - 2.2 * offset),
                fontsize=9, color="#374151", zorder=5,
            )

        ax.plot([], [], color=direct_color, linewidth=2,
                label="Direct shear (Vx, Vy)")
        ax.plot([], [], color=torsion_color, linewidth=2,
                label="Torsion shear (Tx, Ty)")
        ax.plot([], [], color="#9ca3af", linewidth=1, linestyle="--",
                label="Resultant shear")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Bolted union: force distribution")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9)

        if show:
            plt.show()
        return fig

    def __repr__(self):
        return (
            f"BoltedUnion(bolt={self.bolt!r}, n_bolts={self.n_bolts}, "
            f"forces={self.forces}, moments={self.moments})"
        )
