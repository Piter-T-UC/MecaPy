"""Shaft design and analysis module.

Units convention: geometry in mm, torque in N*mm and stresses in MPa
(N/mm^2), consistent with the other element modules (bolts, gears).
"""

import math

from sympy import Rational, Symbol

from ..base import MechaElement
from ..beams import Beam
from .kt_data import get_kt_groove, get_kt_shoulder_fillet

_X = Symbol("x")  # mecapy.beams.Beam's own sympy variable, matched here for substitution


class Shaft(MechaElement):
    """
    Shaft design and analysis.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement` and adds a specialized torsional
    shear-stress calculation for solid circular shafts.

    Attributes:
        diameter (float): Shaft diameter in mm. Settable; changing it
            updates :attr:`polar_moment` and :meth:`torsional_stress`
            automatically.
        length (float): Shaft length in mm. Settable.
        material (str): Material type.
        grooves (list): Grooves attached via :meth:`add_groove`, each a
            ``[number, (position, diameter, radii, width), kt_axial,
            kt_bending, kt_torsion]`` row.
    """

    def __init__(self, diameter, length, material="steel", name=None):
        """
        Initialize a Shaft object.

        Args:
            diameter (float): Shaft diameter in mm. Must be strictly
                positive.
            length (float): Shaft length in mm. Must be strictly
                positive.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the shaft.

        Raises:
            ValueError: If ``diameter`` or ``length`` is not strictly
                positive.
        """
        super().__init__(name=name, material=material)
        self.diameter = diameter  # routed through the validating setter below
        self.length = length
        self.grooves = []
        self._supports = []
        self._point_loads = []

    # ---- Settable primary inputs ----

    @property
    def diameter(self):
        """float: Shaft diameter in mm."""
        return self._diameter

    @diameter.setter
    def diameter(self, value):
        if value <= 0:
            raise ValueError("Shaft diameter must be strictly positive")
        self._diameter = value

    @property
    def length(self):
        """float: Shaft length in mm."""
        return self._length

    @length.setter
    def length(self, value):
        if value <= 0:
            raise ValueError("Shaft length must be strictly positive")
        self._length = value

    # ---- Derived geometry (always recomputed, never cached) ----

    @property
    def polar_moment(self):
        """float: Polar second moment of area J = pi * d^4 / 32 (mm^4)."""
        return math.pi * self.diameter ** 4 / 32

    @property
    def second_moment(self):
        """float: Bending second moment of area I = pi * d^4 / 64 (mm^4)."""
        return math.pi * self.diameter ** 4 / 64

    def torsional_stress(self, torque):
        """
        Calculate the maximum torsional shear stress for a solid shaft.

        Uses ``tau = T * r / J`` with ``r = d / 2`` and
        ``J = pi * d^4 / 32``.

        Args:
            torque (float): Applied torque (N*mm for a diameter in mm).

        Returns:
            float: Maximum surface shear stress (MPa for a diameter in mm
            and a torque in N*mm).
        """
        radius = self.diameter / 2
        return torque * radius / self.polar_moment

    # ---- Loading (supports, forces, shear/moment diagrams) ----
    #
    # mx (torque) has its own single-value torsional_stress(torque) rather
    # than a per-x diagram. fx (axial) and fy/fz/my/mz (bending) both feed
    # stress_profile()/plot(), combined via von Mises — see
    # _axial_force_at for the simplification axial force relies on.

    def add_support(self, location, kind="pin"):
        """
        Add a support (bearing) at an axial location, restraining
        deflection in both bending planes.

        Args:
            location (float): Axial position of the support, in mm.
            kind (str): "pin", "roller", or "fixed" (default: "pin").

        Returns:
            Shaft: self, so calls can be chained.

        Raises:
            ValueError: If location is outside [0, length] or kind is
                unrecognized.
        """
        if not 0 <= location <= self.length:
            raise ValueError(f"Support at location={location}mm is outside the shaft's length")
        if kind not in ("pin", "roller", "fixed"):
            raise ValueError("kind must be 'pin', 'roller', or 'fixed'")
        self._supports.append((location, kind))
        return self

    def forces(self, loc=0, fx=0, fy=0, fz=0, mx=0, my=0, mz=0):
        """
        Apply a point load/moment at an axial location. Call again for
        each additional load point; loads accumulate.

        Args:
            loc (float): Axial position, in mm (default: 0).
            fx (float): Axial force, N (default: 0).
            fy (float): Transverse force in y, N (default: 0).
            fz (float): Transverse force in z, N (default: 0).
            mx (float): Torque about x, N*mm (default: 0).
            my (float): Bending moment about y, N*mm (default: 0).
            mz (float): Bending moment about z, N*mm (default: 0).

        Returns:
            Shaft: self, so calls can be chained.

        Raises:
            ValueError: If loc is outside [0, length].
        """
        if not 0 <= loc <= self.length:
            raise ValueError(f"Load at loc={loc}mm is outside the shaft's length")
        self._point_loads.append((loc, fx, fy, fz, mx, my, mz))
        return self

    def _build_beam(self, plane):
        """Build a fresh mecapy.beams.Beam for one bending plane from the
        stored supports/loads — the only place that touches Beam's API, so
        Beam can change shape later without rippling through Shaft.

        Positions are converted to exact sympy.Rational before reaching
        Beam/sympy: sympy's solve_for_reaction_loads silently breaks (an
        IndexError deep inside linsolve) when the beam length or a
        support/load location is a plain Python float rather than an
        exact number — confirmed on this sympy version. Magnitudes
        (forces/moments) don't need this; only positions do.
        """
        beam = Beam(Rational(self.length), material=self.material, second_moment=self.second_moment)
        for location, kind in self._supports:
            beam.add_support(Rational(location), kind)
        # row = (loc, fx, fy, fz, mx, my, mz); "y" plane uses fy+mz, "z" plane uses fz+my
        force_index, moment_index = (2, 6) if plane == "y" else (3, 5)
        for row in self._point_loads:
            loc = Rational(row[0])
            force, moment = row[force_index], row[moment_index]
            if force:
                beam.add_point_load(force, loc)
            if moment:
                beam.add_moment(moment, loc)
        return beam

    def shear_force(self):
        """
        Shear force diagrams in both bending planes.

        Returns:
            dict: ``{"y": Vy(x), "z": Vz(x)}``, symbolic expressions in
            ``x`` (same convention as ``mecapy.beams.Beam.shear_force``) —
            substitute ``x`` for a numeric value.
        """
        return {"y": self._build_beam("y").shear_force(), "z": self._build_beam("z").shear_force()}

    def bending_moment(self):
        """
        Bending moment diagrams in both planes.

        Returns:
            dict: ``{"y": Mz(x), "z": My(x)}`` — "y" is the moment produced
            by y-direction forces/z-moments (bending about z, in the
            xy-plane); "z" is produced by z-direction forces/y-moments
            (bending about y, in the xz-plane). Symbolic expressions in
            ``x`` (same convention as ``mecapy.beams.Beam.bending_moment``)
            — substitute ``x`` for a numeric value.
        """
        return {
            "y": self._build_beam("y").bending_moment(),
            "z": self._build_beam("z").bending_moment(),
        }

    # ---- Grooves ----

    def add_groove(self, position, diameter, radii, a=0):
        """
        Attach a local groove (stress riser) to this shaft, precomputing
        its stress-concentration factor for every loading type.

        Args:
            position (float): Axial distance from this shaft's start to
                the groove's center, in mm.
            diameter (float): Groove root diameter, in mm. Must be
                strictly positive and smaller than this shaft's diameter.
            radii (float): Corner/blend radius, in mm. For a round groove
                this doubles as the semicircle radius, so the groove's
                width is ``2 * radii``; for a flat groove it's the fillet
                radius at each corner and must be at most half of ``a``.
            a (float): 0 for a round (semicircular) groove; > 0 gives a
                flat-bottom groove whose flat length is ``a`` (default: 0).

        Returns:
            Shaft: self, so calls can be chained.

        Raises:
            ValueError: If any dimension is invalid, or the groove
                doesn't fit within this shaft's length.
        """
        if diameter <= 0 or diameter >= self.diameter:
            raise ValueError("Groove diameter must be positive and smaller than the shaft diameter")
        if radii <= 0:
            raise ValueError("Groove radii must be strictly positive")
        if a < 0:
            raise ValueError("a must be non-negative")
        if a > 0 and radii > a / 2:
            raise ValueError("Groove radii cannot exceed half of a (the flat width)")

        width = 2 * radii if a == 0 else a
        start, end = position - width / 2, position + width / 2
        if start < 0 or end > self.length:
            raise ValueError(
                f"Groove at position={position}mm does not fit within the shaft's length"
            )

        get_kt = get_kt_groove if a == 0 else get_kt_shoulder_fillet
        # ponytail: Kt is computed once here and cached in the row instead
        # of as a property, so it goes stale if self.diameter changes
        # afterward. Upgrade path: store only the raw geometry and
        # recompute Kt in a property, if that staleness ever bites.
        kt_axial = get_kt(self.diameter, diameter, radii, loading="axial")
        kt_bending = get_kt(self.diameter, diameter, radii, loading="bending")
        kt_torsion = get_kt(self.diameter, diameter, radii, loading="torsion")

        number = len(self.grooves) + 1
        row = [number, (position, diameter, radii, width), kt_axial, kt_bending, kt_torsion]
        self.grooves.append(row)
        return self

    def _diameter_at(self, x):
        """Local diameter at axial position x (mm), dipping to a groove's
        root diameter inside its span."""
        for row in self.grooves:
            gx, gd, _, gw = row[1]
            if gx - gw / 2 <= x <= gx + gw / 2:
                return gd
        return self.diameter

    def _torsional_stress_at(self, diameter, torque):
        return Shaft(diameter=diameter, length=1.0, material=self.material).torsional_stress(torque)

    def _bending_stress_at(self, x, diameter, mz_expr, my_expr):
        """Bending stress at x from the resultant of both planes' moments,
        sigma = M*c/I on the local (possibly grooved) diameter."""
        moment = math.hypot(float(mz_expr.subs(_X, x)), float(my_expr.subs(_X, x)))
        return moment * (diameter / 2) / (math.pi * diameter ** 4 / 64)

    def _axial_force_at(self, x):
        """Internal axial force at x: running sum of every applied fx at
        or before x.

        # ponytail: no reaction is solved for — this only balances if the
        # applied fx loads already sum to zero, or x=0 is effectively the
        # shaft's free (axially unsupported) end, matching a bearing
        # arrangement with one locating (thrust) support at/near x=0 and
        # one floating support elsewhere. Upgrade path: let add_support
        # flag which support is the thrust bearing and solve for its
        # reaction, if a case with the thrust bearing elsewhere comes up.
        """
        return sum(row[1] for row in self._point_loads if row[0] <= x)

    def _axial_stress_at(self, x, diameter):
        """Axial stress sigma = N(x) / A at the local (possibly grooved)
        diameter. Signed: positive tension, negative compression."""
        return self._axial_force_at(x) / (math.pi * diameter ** 2 / 4)

    # ---- Stress along the shaft ----

    def stress_profile(self, torque=0, n=100):
        """
        Sample the combined von Mises stress along this shaft — torsion
        (from ``torque``), bending (resultant of both planes), and axial
        (running sum of applied ``fx``, see ``_axial_force_at``) — plus
        the Kt-amplified peak at each groove. Bending and axial are normal
        stresses and simply added (worst-case superposition, matching
        Shigley's static combined-loading equations); torsion combines via
        ``sigma_vm = sqrt((sigma_bending + sigma_axial)^2 + 3*tau^2)``, the
        same convention as ``PowerScrew.von_mises_stress`` elsewhere in
        this codebase.

        Args:
            torque (float): Applied torque, N*mm, assumed constant along
                the shaft (default: 0).
            n (int): Number of points sampling the nominal-stress curve
                (>= 2).

        Returns:
            dict: ``{"x": [mm, ...], "nominal_stress": [MPa, ...],
            "peaks": [{"x", "diameter", "kt_torsion", "kt_bending",
            "kt_axial", "kt", "stress", "number"}, ...]}``, peaks sorted
            by axial position. ``kt`` is the *effective* concentration on
            the combined stress (``stress`` divided by the unconcentrated
            von Mises stress at that point) — a single number for display,
            even though each loading carries its own Kt.

        Raises:
            ValueError: If n < 2.
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        xs = [self.length * i / (n - 1) for i in range(n)]

        moments = self.bending_moment()
        mz_expr, my_expr = moments["y"], moments["z"]

        def combined_stress(x, diameter, kt_torsion=1.0, kt_bending=1.0, kt_axial=1.0):
            tau = kt_torsion * self._torsional_stress_at(diameter, torque)
            sigma_bending = kt_bending * self._bending_stress_at(x, diameter, mz_expr, my_expr)
            sigma_axial = kt_axial * self._axial_stress_at(x, diameter)
            sigma = sigma_bending + sigma_axial
            return math.sqrt(sigma ** 2 + 3 * tau ** 2)

        nominal = [combined_stress(x, self._diameter_at(x)) for x in xs]

        peaks = []
        for number, (gx, gd, _radii, _gw), kt_axial, kt_bending, kt_torsion in self.grooves:
            unconcentrated = combined_stress(gx, gd)
            stress = combined_stress(gx, gd, kt_torsion, kt_bending, kt_axial)
            effective_kt = (
                stress / unconcentrated if unconcentrated else max(kt_torsion, kt_bending, kt_axial)
            )
            peaks.append({
                "x": gx, "diameter": gd, "kt_torsion": kt_torsion, "kt_bending": kt_bending,
                "kt_axial": kt_axial, "kt": effective_kt,
                "stress": stress, "number": number,
            })
        peaks.sort(key=lambda p: p["x"])
        return {"x": xs, "nominal_stress": nominal, "peaks": peaks}

    # ---- Visualization ----

    def _outline(self, n_arc=12):
        """Local (x, r) polyline points for this shaft's outline: a
        straight cylinder with a groove dip (sine-shaped for "round",
        flat-bottomed for "flat") — a schematic drawing, not exact
        groove geometry."""
        xs, rs = [], []

        def add(x, r):
            xs.append(x)
            rs.append(r)

        r_seg = self.diameter / 2
        add(0.0, r_seg)
        for _number, (gx, gd, radii, gw), *_ in sorted(self.grooves, key=lambda row: row[1][0]):
            r_groove = gd / 2
            gx0, gx1 = gx - gw / 2, gx + gw / 2
            if gw == 2 * radii:  # round groove (width derived from radii in add_groove)
                add(gx0, r_seg)
                for t in range(1, n_arc):
                    frac = t / n_arc
                    dip = r_seg - (r_seg - r_groove) * math.sin(frac * math.pi)
                    add(gx0 + frac * (gx1 - gx0), dip)
                add(gx1, r_seg)
            else:  # flat groove
                taper = min(radii, gw / 2)
                add(gx0, r_seg)
                add(gx0 + taper, r_groove)
                add(gx1 - taper, r_groove)
                add(gx1, r_seg)
        add(self.length, r_seg)
        return xs, rs

    def plot(self, torque=0, n=100, show=True, ax=None):
        """
        Draw this shaft's outline (diameter and grooves, to scale) with
        the combined von Mises stress (torsion + bending, see
        :meth:`stress_profile`) along its length plotted beneath it on a
        shared x-axis.

        Args:
            torque (float): Applied torque, N*mm, used for the stress
                curve (default: 0).
            n (int): Number of points sampling the nominal-stress curve.
            show (bool): Call ``plt.show()`` (default: True). Pass False
                when embedding or testing.
            ax (tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]):
                Existing ``(profile_ax, stress_ax)`` pair to draw into.
                If None, a new figure with two stacked subplots is
                created.

        Returns:
            matplotlib.figure.Figure: The figure containing both plots.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot; install it with 'pip install matplotlib'"
            )

        if ax is None:
            fig, (ax_profile, ax_stress) = plt.subplots(
                2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
            )
        else:
            ax_profile, ax_stress = ax
            fig = ax_profile.figure

        outline_x, outline_r = self._outline()
        outline_r_neg = [-r for r in outline_r]
        ax_profile.plot(outline_x, outline_r, color="#1f2937", linewidth=1.5)
        ax_profile.plot(outline_x, outline_r_neg, color="#1f2937", linewidth=1.5)
        ax_profile.fill_between(outline_x, outline_r, outline_r_neg, color="#e5e7eb", zorder=0)
        ax_profile.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
        ax_profile.set_ylabel("radius [mm]")
        ax_profile.set_title(self.name or "Shaft")
        ax_profile.set_aspect("equal", adjustable="datalim")

        profile = self.stress_profile(torque, n=n)
        ax_stress.plot(
            profile["x"], profile["nominal_stress"],
            color="#3c25eb", linewidth=1.5, label="nominal von Mises stress",
        )
        for p in profile["peaks"]:
            ax_stress.plot(p["x"], p["stress"], "o", color="#d90b0b", zorder=3)
            label = (
                f"Kt={p['kt']:.2f} (t={p['kt_torsion']:.2f}, b={p['kt_bending']:.2f}, "
                f"a={p['kt_axial']:.2f})\n{p['stress']:.1f} MPa"
            )
            ax_stress.annotate(
                label,
                (p["x"], p["stress"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color="#d90b0b",
            )
        ax_stress.set_xlabel("axial position [mm]")
        ax_stress.set_ylabel("von Mises stress [MPa]")
        ax_stress.grid(True, color="#e5e7eb", linewidth=0.8)
        ax_stress.set_axisbelow(True)
        ax_stress.legend(loc="best", fontsize=9)

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def __repr__(self):
        return (
            f"Shaft(diameter={self.diameter}mm, length={self.length}mm, material={self.material!r})"
        )
