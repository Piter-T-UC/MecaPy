"""Shaft design and analysis module.

Units convention: geometry in mm, torque in N*mm and stresses in MPa
(N/mm^2), consistent with the other element modules (bolts, gears).
"""

import math

from ..base import MechaElement
from .kt_data import get_kt_groove, get_kt_shoulder_fillet


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

    # ---- Stress along the shaft ----

    def stress_profile(self, torque, n=100):
        """
        Sample nominal torsional stress along this shaft, plus the
        Kt-amplified peak stress at each groove.

        Args:
            torque (float): Applied torque, N*mm, assumed constant along
                the shaft.
            n (int): Number of points sampling the nominal-stress curve
                (>= 2).

        Returns:
            dict: ``{"x": [mm, ...], "nominal_stress": [MPa, ...],
            "peaks": [{"x", "diameter", "kt", "stress", "number"}, ...]}``,
            peaks sorted by axial position.

        Raises:
            ValueError: If n < 2.
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        xs = [self.length * i / (n - 1) for i in range(n)]
        nominal = [self._torsional_stress_at(self._diameter_at(x), torque) for x in xs]

        peaks = []
        for number, (gx, gd, _radii, _gw), _kt_axial, _kt_bending, kt_torsion in self.grooves:
            nominal_stress = self._torsional_stress_at(gd, torque)
            peaks.append({
                "x": gx, "diameter": gd, "kt": kt_torsion,
                "stress": kt_torsion * nominal_stress, "number": number,
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

    def plot(self, torque, n=100, show=True, ax=None):
        """
        Draw this shaft's outline (diameter and grooves, to scale) with
        the torsional stress along its length plotted beneath it on a
        shared x-axis.

        Args:
            torque (float): Applied torque, N*mm, used for the stress
                curve.
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
            color="#3c25eb", linewidth=1.5, label="nominal stress",
        )
        for p in profile["peaks"]:
            ax_stress.plot(p["x"], p["stress"], "o", color="#d90b0b", zorder=3)
            ax_stress.annotate(
                f"Kt={p['kt']:.2f}\n{p['stress']:.1f} MPa",
                (p["x"], p["stress"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color="#d90b0b",
            )
        ax_stress.set_xlabel("axial position [mm]")
        ax_stress.set_ylabel("torsional stress [MPa]")
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
