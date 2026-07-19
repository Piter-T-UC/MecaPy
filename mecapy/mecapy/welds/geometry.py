"""Weld path geometry for the "weld as a line" method.

Each weld is laid out along a path — a straight :class:`WeldLine` or a
full :class:`WeldCircle`. Following Blodgett's "treat the weld as a line"
method (Shigley's *Mechanical Engineering Design*, Ch. 9), a path exposes
its length, centroid and *unit* second moments of area (the weld treated
as a line of unit throat, so the moments have units of mm^3). The actual
section properties are obtained by multiplying by the throat thickness.

Units: coordinates and lengths in mm; unit second moments in mm^3.
"""

import math


class WeldPath:
    """Abstract weld path. Subclasses implement the geometry."""

    @property
    def length(self):
        """float: Length of the weld path in mm."""
        raise NotImplementedError

    @property
    def centroid(self):
        """tuple: Centroid (x, y) of the path in mm."""
        raise NotImplementedError

    def unit_inertia(self):
        """
        Unit second moments of area about the path's own centroid.

        Returns:
            tuple: ``(Ix, Iy)`` in mm^3 (weld treated as a line of unit
            throat), taken about centroidal axes parallel to x and y.
        """
        raise NotImplementedError

    def sample_points(self, n):
        """
        Sample points along the path (for stress evaluation and plotting).

        Args:
            n (int): Number of points (>= 2).

        Returns:
            list: ``[(x, y), ...]`` in mm.
        """
        raise NotImplementedError


class WeldLine(WeldPath):
    """
    Straight weld segment between two points.

    Attributes:
        start (tuple): Start point (x, y) in mm.
        end (tuple): End point (x, y) in mm.
    """

    def __init__(self, start, end):
        """
        Initialize a straight weld segment.

        Args:
            start (tuple): Start point (x, y) in mm.
            end (tuple): End point (x, y) in mm.

        Raises:
            ValueError: If the two points coincide (zero length).
        """
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        if self.length == 0:
            raise ValueError("Weld line must have two distinct endpoints")

    @property
    def length(self):
        """float: Segment length in mm."""
        return math.hypot(self.end[0] - self.start[0],
                          self.end[1] - self.start[1])

    @property
    def centroid(self):
        """tuple: Midpoint (x, y) of the segment in mm."""
        return ((self.start[0] + self.end[0]) / 2,
                (self.start[1] + self.end[1]) / 2)

    def unit_inertia(self):
        """Unit second moments (Ix, Iy) in mm^3 about the midpoint.

        For a line of length L with end-to-end offsets dx, dy:
        ``Ix = L * dy^2 / 12``, ``Iy = L * dx^2 / 12``. A vertical weld
        of length d gives ``Ix = d^3 / 12`` (Shigley Table 9-2).
        """
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = self.length
        return (length * dy ** 2 / 12.0, length * dx ** 2 / 12.0)

    def sample_points(self, n):
        """Return ``n`` points evenly spaced from start to end."""
        if n < 2:
            raise ValueError("Need at least 2 sample points")
        x0, y0 = self.start
        x1, y1 = self.end
        return [
            (x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1))
            for i in range(n)
        ]

    def __repr__(self):
        return f"WeldLine(start={self.start}, end={self.end})"


class WeldCircle(WeldPath):
    """
    Full circular weld.

    Attributes:
        center (tuple): Center (x, y) in mm.
        radius (float): Radius in mm.
    """

    def __init__(self, center, radius):
        """
        Initialize a full circular weld.

        Args:
            center (tuple): Center (x, y) in mm.
            radius (float): Radius in mm (> 0).

        Raises:
            ValueError: If ``radius`` is not strictly positive.
        """
        if radius <= 0:
            raise ValueError("Weld circle radius must be strictly positive")
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)

    @property
    def length(self):
        """float: Circumference 2*pi*r in mm."""
        return 2 * math.pi * self.radius

    @property
    def centroid(self):
        """tuple: Circle center (x, y) in mm."""
        return self.center

    def unit_inertia(self):
        """Unit second moments (Ix, Iy) in mm^3 about the center.

        For a full circular line of radius r, ``Ix = Iy = pi * r^3``, so
        the unit polar moment is ``Ju = 2 * pi * r^3`` (Shigley Table
        9-1/9-2).
        """
        i = math.pi * self.radius ** 3
        return (i, i)

    def sample_points(self, n):
        """Return ``n`` points evenly spaced around the circumference."""
        if n < 2:
            raise ValueError("Need at least 2 sample points")
        cx, cy = self.center
        return [
            (cx + self.radius * math.cos(2 * math.pi * i / n),
             cy + self.radius * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]

    def __repr__(self):
        return f"WeldCircle(center={self.center}, radius={self.radius})"
