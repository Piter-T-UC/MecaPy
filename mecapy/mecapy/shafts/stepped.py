"""Stepped (multi-diameter) shaft module.

Units convention: geometry in mm, torque in N*mm and stresses in MPa
(N/mm^2), consistent with :mod:`mecapy.shafts.shaft`.
"""

from ..base import MechaElement
from .kt_data import get_kt_shoulder_fillet
from .shaft import Shaft


class SteppedShaft(MechaElement):
    """
    A stepped shaft: two or more coaxial Shaft segments fused end-to-end
    by chaining :meth:`add_shaft`, e.g. a lathe-turned shaft with
    shoulders.

    Grooves belong to each individual Shaft segment (see
    :meth:`~mecapy.shafts.shaft.Shaft.add_groove`); this class only fuses
    segments together and owns the fillet at each transition between
    them.

    Attributes:
        segments (list[Shaft]): Segments in end-to-end order.
        fillets (list): One row per diameter transition, aligned so
            ``fillets[i]`` sits between ``segments[i]`` and
            ``segments[i + 1]``: ``[number, (D, d, radius), kt_axial,
            kt_bending, kt_torsion]``.
        material (str): Material of the base shaft.
    """

    def __init__(self, base_shaft, name=None):
        """
        Initialize a SteppedShaft from a single starting Shaft. Grow it
        with :meth:`add_shaft`.

        Args:
            base_shaft (Shaft): The first segment.
            name (str): Optional identifier.

        Raises:
            ValueError: If base_shaft is not a Shaft.
        """
        if not isinstance(base_shaft, Shaft):
            raise ValueError("base_shaft must be a Shaft instance")
        super().__init__(name=name, material=base_shaft.material)
        self.segments = [base_shaft]
        self.fillets = []

    # ---- Growing the shaft ----

    def add_shaft(self, shaft, transition, radius, side="right"):
        """
        Fuse another Shaft segment onto this stepped shaft.

        Args:
            shaft (Shaft): Segment to add.
            transition (str): Must be ``"fillet"`` — the only supported
                transition kind between two fused shafts today.
            radius (float): Fillet radius at the new transition, in mm.
                Must be strictly positive and smaller than half of the
                smaller adjoining diameter.
            side (str): ``"right"`` (default) appends after the last
                segment; ``"left"`` prepends before the first segment.

        Returns:
            SteppedShaft: self, so calls can be chained.

        Raises:
            ValueError: If transition isn't "fillet", side isn't
                "left"/"right", shaft isn't a Shaft, or radius doesn't
                fit the adjoining diameters.
        """
        if transition != "fillet":
            raise ValueError("Only 'fillet' transitions are supported")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if not isinstance(shaft, Shaft):
            raise ValueError("shaft must be a Shaft instance")

        neighbor = self.segments[-1] if side == "right" else self.segments[0]
        D = max(neighbor.diameter, shaft.diameter)
        d = min(neighbor.diameter, shaft.diameter)
        if radius <= 0:
            raise ValueError("Fillet radius must be strictly positive")
        if radius >= d / 2:
            raise ValueError(f"Fillet radius is too large for the adjoining {d}mm diameter")

        number = len(self.fillets) + 1
        row = [
            number, (D, d, radius),
            get_kt_shoulder_fillet(D, d, radius, loading="axial"),
            get_kt_shoulder_fillet(D, d, radius, loading="bending"),
            get_kt_shoulder_fillet(D, d, radius, loading="torsion"),
        ]

        if side == "right":
            self.segments.append(shaft)
            self.fillets.append(row)
        else:
            self.segments.insert(0, shaft)
            self.fillets.insert(0, row)
        return self

    # ---- Derived geometry (always recomputed, never cached) ----

    @property
    def boundaries(self):
        """list[float]: Cumulative axial position (mm) of each segment's
        start, ending with the total length."""
        cum = [0.0]
        for s in self.segments:
            cum.append(cum[-1] + s.length)
        return cum

    @property
    def total_length(self):
        """float: Total axial length of the fused shaft, in mm."""
        return sum(s.length for s in self.segments)

    def _segment_index_at(self, x):
        boundaries = self.boundaries
        for i in range(len(self.segments)):
            if boundaries[i] <= x <= boundaries[i + 1]:
                return i
        raise ValueError(f"x={x}mm is outside the shaft's length [0, {self.total_length}mm]")

    def diameter_at(self, x):
        """
        Nominal shaft diameter at axial position x, accounting for any
        groove on the owning segment whose span covers x.

        Args:
            x (float): Axial position, mm, from the shaft's start.

        Returns:
            float: Local diameter, mm.

        Raises:
            ValueError: If x is outside [0, total_length].
        """
        if not 0 <= x <= self.total_length:
            raise ValueError(f"x={x}mm is outside the shaft's length [0, {self.total_length}mm]")
        idx = self._segment_index_at(x)
        return self.segments[idx]._diameter_at(x - self.boundaries[idx])

    # ---- Stress ----

    def stress_profile(self, torque, n=200):
        """
        Sample nominal torsional stress along the fused shaft by
        stitching together each segment's own
        :meth:`~mecapy.shafts.shaft.Shaft.stress_profile`, plus one
        Kt-amplified peak per fillet.

        Args:
            torque (float): Applied torque, N*mm, assumed constant along
                the shaft.
            n (int): Total number of points sampling the nominal-stress
                curve, split evenly across segments (>= 2).

        Returns:
            dict: ``{"x": [mm, ...], "nominal_stress": [MPa, ...],
            "peaks": [{"x", "diameter", "kt", "stress", "kind",
            "number"}, ...]}``, peaks sorted by axial position; ``kind``
            is ``"groove"`` or ``"fillet"``.

        Raises:
            ValueError: If n < 2.
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        per_segment = max(2, n // len(self.segments))
        boundaries = self.boundaries

        xs, nominal, peaks = [], [], []
        for i, segment in enumerate(self.segments):
            local = segment.stress_profile(torque, n=per_segment)
            offset = boundaries[i]
            xs.extend(x + offset for x in local["x"])
            nominal.extend(local["nominal_stress"])
            for p in local["peaks"]:
                peaks.append({**p, "x": p["x"] + offset, "kind": "groove"})

        for i, fillet in enumerate(self.fillets):
            number, (_D, d, _radius), _kt_axial, _kt_bending, kt_torsion = fillet
            smaller = self.segments[i] if self.segments[i].diameter == d else self.segments[i + 1]
            nominal_stress = smaller.torsional_stress(torque)
            peaks.append({
                "x": boundaries[i + 1], "diameter": d, "kt": kt_torsion,
                "stress": kt_torsion * nominal_stress, "kind": "fillet", "number": number,
            })

        peaks.sort(key=lambda p: p["x"])
        return {"x": xs, "nominal_stress": nominal, "peaks": peaks}

    # ---- Visualization ----

    def _stitched_outline(self):
        """Concatenate each segment's own outline (shifted to its axial
        offset), pulling the endpoints at each transition inward by its
        fillet radius for a linear chamfer visual — grouping is this
        class's job, drawing a single segment is Shaft's."""
        xs, rs = [], []
        boundaries = self.boundaries
        for i, segment in enumerate(self.segments):
            seg_x, seg_r = segment._outline()
            seg_x = [x + boundaries[i] for x in seg_x]
            if i > 0:
                radius = self.fillets[i - 1][1][2]
                seg_x[0] += min(radius, segment.length / 2)
            if i < len(self.segments) - 1:
                radius = self.fillets[i][1][2]
                seg_x[-1] -= min(radius, segment.length / 2)
            xs.extend(seg_x)
            rs.extend(seg_r)
        return xs, rs

    def plot(self, torque, n=200, show=True, ax=None):
        """
        Draw the stepped shaft's outline (diameter changes, fillets, and
        grooves, to scale) with the torsional stress along its length
        plotted beneath it on a shared x-axis.

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

        outline_x, outline_r = self._stitched_outline()
        outline_r_neg = [-r for r in outline_r]
        ax_profile.plot(outline_x, outline_r, color="#1f2937", linewidth=1.5)
        ax_profile.plot(outline_x, outline_r_neg, color="#1f2937", linewidth=1.5)
        ax_profile.fill_between(outline_x, outline_r, outline_r_neg, color="#e5e7eb", zorder=0)
        ax_profile.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
        ax_profile.set_ylabel("radius [mm]")
        ax_profile.set_title(self.name or "SteppedShaft")
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
            f"SteppedShaft(segments={len(self.segments)}, "
            f"total_length={self.total_length}mm, material={self.material!r})"
        )
