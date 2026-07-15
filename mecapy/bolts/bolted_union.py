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
            "shear_magnitude": float, "axial": float}}`` with forces
            in N.

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
            fsx = fx / n
            fsy = fy / n
            if mz != 0:
                fsx += -mz * dy / sum_r2
                fsy += mz * dx / sum_r2
            axial = fz / n
            if mx != 0:
                axial += mx * dy / sum_dy2
            if my != 0:
                axial += -my * dx / sum_dx2
            result[number] = {
                "shear": (fsx, fsy),
                "shear_magnitude": sqrt(fsx ** 2 + fsy ** 2),
                "axial": axial,
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

        Each bolt is drawn at its position; its shear force is drawn as
        an arrow pointing in the shear direction, and its axial force is
        annotated as a magnitude next to the bolt. The group centroid is
        marked with a cross.

        Args:
            scale (float): Arrow scale in mm per N. If None, the longest
                shear arrow is auto-scaled to about 20% of the plot span.
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
            max_shear = max(entry["shear_magnitude"] for entry in forces.values())
            scale = 0.2 * span / max_shear if max_shear > 0 else 1.0

        ax.scatter(xs, ys, s=80, color="#374151", zorder=3, label="Bolts")
        x_bar, y_bar = self.centroid
        ax.plot(x_bar, y_bar, "+", color="#6b7280", markersize=12,
                markeredgewidth=2, zorder=2, label="Centroid")

        offset = 0.03 * span
        for row in self.positions:
            number, x, y = row
            entry = forces[number]
            fsx, fsy = entry["shear"]
            if entry["shear_magnitude"] > 0:
                ax.annotate(
                    "", xy=(x + fsx * scale, y + fsy * scale), xytext=(x, y),
                    arrowprops={"arrowstyle": "-|>", "color": "#2563eb",
                                "linewidth": 2, "mutation_scale": 16},
                    zorder=4,
                )
            ax.annotate(
                f"#{number}\nN = {entry['axial']:.0f} N",
                xy=(x, y), xytext=(x + offset, y + offset),
                fontsize=9, color="#374151", zorder=5,
            )

        ax.plot([], [], color="#2563eb", linewidth=2, label="Shear force")
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
