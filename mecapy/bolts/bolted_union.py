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

from math import sqrt

from ..base import MechaElement
from .bolt import Bolt


class BoltedUnion(MechaElement):
    """
    Bolted union (bolt group) under combined loading.

    All bolts are identical (same :class:`Bolt` instance), so the
    elastic distribution depends only on the bolt positions: direct
    shear and axial force split equally, torsion produces shear
    proportional to the distance from the centroid, and bending
    produces axial force proportional to the distance from the
    centroidal axis.

    Attributes:
        bolt (Bolt): The bolt used at every position.
        positions (list): List of ``[bolt_number, x, y]`` rows in mm.
        forces (tuple): Applied forces (Fx, Fy, Fz) in N.
        moments (tuple): Applied moments (Mx, My, Mz) in N*mm.
    """

    def __init__(self, bolt, positions, forces=(0, 0, 0), moments=(0, 0, 0), name=None):
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
            name (str): Optional identifier for the union.

        Raises:
            ValueError: If ``bolt`` is not a Bolt, a position row does
                not have exactly 3 entries, bolt numbers repeat, or the
                force/moment vectors do not have 3 components.
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

    # ---- Geometry ----

    @property
    def n_bolts(self):
        """int: Number of bolts in the group."""
        return len(self.positions)

    @property
    def centroid(self):
        """tuple: Centroid (x, y) of the bolt group in mm.

        Simple average of the positions — all bolts have equal area.
        """
        n = self.n_bolts
        x_bar = sum(row[1] for row in self.positions) / n
        y_bar = sum(row[2] for row in self.positions) / n
        return (x_bar, y_bar)

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
            scale (float): Arrow scale in mm per N. If None, the largest
                shear component is auto-scaled to about 20% of the plot
                span.
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
        if scale is None:
            max_component = max(
                (abs(c) for entry in forces.values()
                 for pair in (entry["shear_direct"], entry["shear_torsion"])
                 for c in pair),
                default=0.0,
            )
            scale = 0.2 * span / max_component if max_component > 0 else 1.0

        ax.scatter(xs, ys, s=80, color="#374151", zorder=3, label="Bolts")
        x_bar, y_bar = self.centroid
        ax.plot(x_bar, y_bar, "+", color="#6b7280", markersize=12,
                markeredgewidth=2, zorder=2, label="Centroid")

        direct_color = "#2563eb"
        torsion_color = "#ea580c"
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
            tip = (base[0] + cx * scale, base[1] + cy * scale)
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
                           (0, 0.5 * offset),
                           (vx_side * 0.3 * offset, 0.5 * offset),
                           "left" if vx_side > 0 else "right")
            draw_component(x, y, 0, vy, "Vy", vy, direct_color,
                           (0.5 * offset, 0), (0.5 * offset, 0.3 * offset), "left")
            draw_component(x, y, tx, 0, "Tx", tx, torsion_color,
                           (0, -0.5 * offset),
                           (tx_side * 0.3 * offset, -0.5 * offset),
                           "left" if tx_side > 0 else "right")
            draw_component(x, y, 0, ty, "Ty", ty, torsion_color,
                           (-0.5 * offset, 0), (-0.5 * offset, 0.3 * offset), "right")
            fsx, fsy = entry["shear"]
            if entry["shear_magnitude"] > 0:
                ax.annotate(
                    "", xy=(x + fsx * scale, y + fsy * scale), xytext=(x, y),
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
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.margins(0.25)
        ax.legend(loc="best", fontsize=9)

        if show:
            plt.show()
        return fig

    def __repr__(self):
        return (
            f"BoltedUnion(bolt={self.bolt!r}, n_bolts={self.n_bolts}, "
            f"forces={self.forces}, moments={self.moments})"
        )
