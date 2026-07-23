"""Welded union (weld group) analysis module.

Distributes external loads over a group of welds using the elastic
"weld as a line" method (Shigley's *Mechanical Engineering Design*,
Ch. 9). Each weld is treated as a line of throat thickness; the group's
*unit* second and polar moments of area (mm^3) come from the line
geometry, and the throat converts force-per-length into stress. Units:
coordinates in mm, forces in N, moments in N*mm, stresses in MPa.

Load convention (right-handed axes, z out of the joint plane), identical
to :class:`~mecapy.bolts.bolted_union.BoltedUnion`:
    - forces = (Fx, Fy, Fz): Fx, Fy are in-plane shear on the group,
      Fz is the axial (out-of-plane) force.
    - moments = (Mx, My, Mz): Mx and My are bending moments about the
      x and y axes, Mz is the torsion moment about z.
    - All loads act at the centroid of the weld group.
"""

from math import hypot, sqrt

from ..base import MechaElement
from .weld import Weld


class WeldedUnion(MechaElement):
    """
    Welded union (weld group) under combined loading.

    The welds are treated as lines (Shigley Ch. 9): direct shear and
    axial force spread over the total length, torsion produces in-plane
    shear proportional to the distance from the centroid (scaled by the
    unit polar moment), and bending produces a normal stress proportional
    to the distance from the centroidal axis (scaled by the unit second
    moment). The resulting force per unit length is divided by the throat
    to get a stress, and the axial (normal) and shear parts are combined
    with the von Mises criterion.

    Attributes:
        welds (list): The :class:`~mecapy.welds.weld.Weld` runs.
        forces (tuple): Applied forces (Fx, Fy, Fz) in N.
        moments (tuple): Applied moments (Mx, My, Mz) in N*mm.
    """

    def __init__(self, welds, forces=(0, 0, 0), moments=(0, 0, 0),
                 size=None, name=None):
        """
        Initialize a welded union.

        Args:
            welds (list): List of :class:`~mecapy.welds.weld.Weld` runs.
            forces (tuple): Applied forces (Fx, Fy, Fz) in N at the
                group centroid (default: no force).
            moments (tuple): Applied moments (Mx, My, Mz) in N*mm about
                the centroid: Mx, My bending, Mz torsion (default: none).
            size (float): If given, this weld size (mm) is applied to
                every weld, overriding their individual sizes — a quick
                way to size the whole group at once.
            name (str): Optional identifier for the union.

        Raises:
            ValueError: If ``welds`` is empty or holds non-Weld objects,
                or the force/moment vectors do not have 3 components.
        """
        if not welds:
            raise ValueError("welds must contain at least one Weld")
        for w in welds:
            if not isinstance(w, Weld):
                raise ValueError("every item in welds must be a Weld instance")
        if len(forces) != 3:
            raise ValueError("forces must have 3 components (Fx, Fy, Fz)")
        if len(moments) != 3:
            raise ValueError("moments must have 3 components (Mx, My, Mz)")
        super().__init__(name=name, material=welds[0].material)
        self.welds = list(welds)
        self.forces = tuple(float(f) for f in forces)
        self.moments = tuple(float(m) for m in moments)
        if size is not None:
            self.set_size(size)

    # ---- Geometry ----

    @property
    def n_welds(self):
        """int: Number of weld runs in the group."""
        return len(self.welds)

    @property
    def total_length(self):
        """float: Total weld length in mm."""
        return sum(w.length for w in self.welds)

    @property
    def centroid(self):
        """tuple: Length-weighted centroid (x, y) of the group in mm."""
        total = self.total_length
        cx = sum(w.length * w.centroid[0] for w in self.welds) / total
        cy = sum(w.length * w.centroid[1] for w in self.welds) / total
        return (cx, cy)

    @property
    def unit_second_moment(self):
        """tuple: Unit second moments (Iux, Iuy) of the group in mm^3.

        The normal (bending) moments of area about the centroidal x and
        y axes, with the welds treated as lines of unit throat.
        """
        cx, cy = self.centroid
        iux = 0.0
        iuy = 0.0
        for w in self.welds:
            ix, iy = w.unit_inertia_about(cx, cy)
            iux += ix
            iuy += iy
        return (iux, iuy)

    @property
    def unit_polar_moment(self):
        """float: Unit polar moment Ju = Iux + Iuy of the group in mm^3."""
        iux, iuy = self.unit_second_moment
        return iux + iuy

    @property
    def throat_area(self):
        """float: Total throat area of the group in mm^2."""
        return sum(w.throat_area for w in self.welds)

    # ---- Weld size (bulk) ----

    @property
    def size(self):
        """float: The common weld size in mm.

        Raises:
            ValueError: If the welds do not all share a single size.
        """
        sizes = {w.size for w in self.welds}
        if len(sizes) != 1:
            raise ValueError(
                "Welds have mixed sizes; set 'size' to make them uniform"
            )
        return sizes.pop()

    @size.setter
    def size(self, value):
        self.set_size(value)

    def set_size(self, size):
        """
        Set the same weld size on every weld in the group.

        Because the unit section properties (Iux, Iuy, Ju, length) are
        line geometry independent of the throat, resizing only rescales
        the stresses — no geometry is recomputed.

        Args:
            size (float): Weld size in mm (> 0), applied to all welds.

        Raises:
            ValueError: If ``size`` is not strictly positive.
        """
        if size <= 0:
            raise ValueError("Weld size must be strictly positive")
        for w in self.welds:
            w.size = size

    # ---- Load distribution and stress ----

    def _load_per_length(self, dx, dy):
        """Force per unit length (fx, fy, fz) in N/mm at (dx, dy) from C."""
        fx0, fy0, fz0 = self.forces
        mx, my, mz = self.moments
        length = self.total_length
        iux, iuy = self.unit_second_moment
        ju = iux + iuy

        if mz != 0 and ju == 0:
            raise ValueError("Torsion applied but the weld group has no lever arm")
        if mx != 0 and iux == 0:
            raise ValueError("Bending Mx applied but all welds lie on the x axis")
        if my != 0 and iuy == 0:
            raise ValueError("Bending My applied but all welds lie on the y axis")

        fx = fx0 / length
        fy = fy0 / length
        if mz != 0:
            fx += -mz * dy / ju
            fy += mz * dx / ju
        fz = fz0 / length
        if mx != 0:
            fz += mx * dy / iux
        if my != 0:
            fz += -my * dx / iuy
        return (fx, fy, fz)

    def _stress_at(self, x, y, throat):
        """von Mises stress (MPa) at global point (x, y) for a throat."""
        cx, cy = self.centroid
        fx, fy, fz = self._load_per_length(x - cx, y - cy)
        tau = hypot(fx, fy) / throat
        sigma = fz / throat
        return sqrt(sigma ** 2 + 3 * tau ** 2)

    def weld_stresses(self, n=24):
        """
        Evaluate the von Mises stress along every weld.

        Args:
            n (int): Number of sample points per weld (>= 2).

        Returns:
            list: One dict per weld (in input order):
            ``{"points": [(x, y), ...], "stresses": [MPa, ...],
            "max_stress": MPa, "max_point": (x, y)}``.
        """
        results = []
        for w in self.welds:
            throat = w.throat
            points = w.sample_points(n)
            stresses = [self._stress_at(x, y, throat) for x, y in points]
            imax = max(range(len(stresses)), key=lambda i: stresses[i])
            results.append({
                "points": points,
                "stresses": stresses,
                "max_stress": stresses[imax],
                "max_point": points[imax],
            })
        return results

    def max_stress(self, n=24):
        """
        Most stressed point in the group.

        Args:
            n (int): Sample points per weld (>= 2).

        Returns:
            tuple: ``(weld_index, (x, y), sigma_eq)`` — the weld's index
            in the input list, the point in mm, and the von Mises stress
            in MPa.
        """
        best = (0, None, -1.0)
        for i, entry in enumerate(self.weld_stresses(n)):
            if entry["max_stress"] > best[2]:
                best = (i, entry["max_point"], entry["max_stress"])
        return best

    def safety_factors(self, n=24):
        """
        Safety factor of each weld against the electrode throat allowable.

        Compares the weld's peak von Mises stress with the electrode
        allowable ``0.30 * FEXX`` (Shigley Table 9-4). This is the weld-metal
        limit only; see :meth:`base_metal_safety_factors` for the adjacent
        base-metal limit and :meth:`governing_safety_factors` for both.

        Args:
            n (int): Sample points per weld (>= 2).

        Returns:
            dict: ``{weld_index: safety_factor}``; an unloaded weld gets
            ``float("inf")``.
        """
        result = {}
        for i, (w, entry) in enumerate(zip(self.welds, self.weld_stresses(n))):
            peak = entry["max_stress"]
            result[i] = w.allowable_stress / peak if peak > 0 else float("inf")
        return result

    def base_metal_safety_factors(self, n=24):
        """
        Safety factor of each weld against the adjacent base-metal limit.

        The group stresses are computed on the throat; the base metal shears
        on the weld leg, so the base-metal stress is ``peak * throat/size``
        (0.707 of the throat stress for a fillet). It is compared with the
        base-metal allowable ``0.40 * Sy`` (Shigley Table 9-5/9-6).

        Args:
            n (int): Sample points per weld (>= 2).

        Returns:
            dict: ``{weld_index: safety_factor}``; an unloaded weld gets
            ``float("inf")``.
        """
        result = {}
        for i, (w, entry) in enumerate(zip(self.welds, self.weld_stresses(n))):
            peak = entry["max_stress"]
            base_stress = peak * w.base_metal_factor
            result[i] = (w.base_metal_allowable / base_stress
                         if base_stress > 0 else float("inf"))
        return result

    def governing_safety_factors(self, n=24):
        """
        Governing safety factor of each weld: min(weld metal, base metal).

        A welded joint must satisfy both the electrode throat limit
        (:meth:`safety_factors`) and the base-metal limit
        (:meth:`base_metal_safety_factors`); the smaller governs.

        Args:
            n (int): Sample points per weld (>= 2).

        Returns:
            dict: ``{weld_index: safety_factor}``.
        """
        weld = self.safety_factors(n)
        base = self.base_metal_safety_factors(n)
        return {i: min(weld[i], base[i]) for i in weld}

    # ---- Sizing ----

    def required_size(self, n=24, safety_factor=1.0, apply=False):
        """
        Weld size (mm) needed to carry the applied loads.

        Uses the "weld as a line" method: the force per unit length at
        any point depends only on the group's line geometry — total
        length (perimeter), unit second moments (Iux, Iuy) and unit
        polar moment Ju — and the applied forces and moments, never on
        the throat. The required throat is therefore closed-form:

            q_eq = sqrt(fz^2 + 3*(fx^2 + fy^2))   [N/mm]
            throat = safety_factor * max(q_eq) / allowable

        evaluated per weld against its own electrode allowable
        (0.30 * FEXX), converting throat to leg size for fillet welds
        (size = throat / 0.707). The group size is the largest per-weld
        requirement. Works whether or not sizes are currently set.

        Args:
            n (int): Sample points per weld (>= 2).
            safety_factor (float): Design factor against the AISC
                allowable (default 1.0, i.e. stress = allowable).
            apply (bool): If True, set the computed size on every weld
                via :meth:`set_size`.

        Returns:
            float: Required weld size in mm (fillet leg or butt
            thickness, whichever governs).

        Raises:
            ValueError: If ``safety_factor`` is not strictly positive or
                the group carries no load (required size would be zero).
        """
        if safety_factor <= 0:
            raise ValueError("safety_factor must be strictly positive")
        from .weld import FILLET_THROAT_FACTOR

        cx, cy = self.centroid
        best = 0.0
        for w in self.welds:
            q_max = 0.0
            for x, y in w.sample_points(n):
                fx, fy, fz = self._load_per_length(x - cx, y - cy)
                q_eq = sqrt(fz ** 2 + 3 * (fx ** 2 + fy ** 2))
                q_max = max(q_max, q_eq)
            throat = safety_factor * q_max / w.allowable_stress
            size = throat / FILLET_THROAT_FACTOR if w.weld_type == "fillet" else throat
            best = max(best, size)
        if best == 0:
            raise ValueError("No load applied: required weld size is zero")
        if apply:
            self.set_size(best)
        return best

    # ---- Visualization ----

    def plot_distribution(self, n=48, show=True, ax=None):
        """
        Plot the weld group coloured by von Mises stress.

        Each weld path is drawn as a sequence of short segments coloured
        by the local stress (MPa) with a colorbar; the group centroid is
        marked with a cross.

        Args:
            n (int): Sample points per weld for the colour map (>= 2).
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
            from matplotlib.collections import LineCollection
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_distribution; "
                "install it with 'pip install matplotlib'"
            )

        data = self.weld_stresses(n)
        segments = []
        seg_values = []
        for entry in data:
            pts = entry["points"]
            stresses = entry["stresses"]
            for j in range(len(pts) - 1):
                segments.append([pts[j], pts[j + 1]])
                seg_values.append((stresses[j] + stresses[j + 1]) / 2)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = ax.figure

        lc = LineCollection(segments, cmap="coolwarm", linewidth=5, zorder=3)
        lc.set_array(seg_values)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label("von Mises stress [MPa]")

        x_bar, y_bar = self.centroid
        ax.plot(x_bar, y_bar, "+", color="#6b7280", markersize=12,
                markeredgewidth=2, zorder=4, label="Centroid")

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Welded union: stress distribution")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.margins(0.25)
        ax.autoscale_view()
        ax.legend(loc="best", fontsize=9)

        if show:
            plt.show()
        return fig

    def __repr__(self):
        return (
            f"WeldedUnion(n_welds={self.n_welds}, "
            f"forces={self.forces}, moments={self.moments})"
        )
