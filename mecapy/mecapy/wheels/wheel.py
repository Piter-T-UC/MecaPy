"""Wheel design and analysis module."""

from ..base import MechaElement


class Wheel(MechaElement):
    """
    Wheel design and analysis.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`.

    Attributes:
        radius (float): Wheel radius in m.
        mass (float): Wheel mass in kg.
        material (str): Material type.
    """

    def __init__(self, radius, material="steel", mass=None, name=None):
        """
        Initialize a Wheel object.

        Args:
            radius (float): Wheel radius in m. Must be strictly positive.
            material (str): Material type (default: "steel").
            mass (float): Optional wheel mass in kg.
            name (str): Optional identifier for the wheel.

        Raises:
            ValueError: If ``radius`` is not strictly positive.
        """
        super().__init__(name=name, material=material)
        if radius <= 0:
            raise ValueError("Radius must be strictly positive")
        self.radius = radius
        self.mass = mass

    @property
    def moment_of_inertia(self):
        """
        float: Moment of inertia about the axis in kg*m^2.

        Modeled as a uniform solid disc, ``I = (1/2) * m * r^2``.
        Subclasses with other geometries (e.g. an annular flywheel)
        override this.

        Raises:
            ValueError: If the wheel mass was not given.
        """
        if self.mass is None:
            raise ValueError("Mass is required to compute the moment of inertia")
        return 0.5 * self.mass * self.radius ** 2

    def kinetic_energy(self, omega):
        """
        Rotational kinetic energy at a given speed.

        ::

            E = (1/2) * I * omega^2

        Args:
            omega (float): Angular velocity in rad/s.

        Returns:
            float: Kinetic energy in J.

        Raises:
            ValueError: If the wheel mass was not given.
        """
        return 0.5 * self.moment_of_inertia * omega ** 2

    def __repr__(self):
        return f"Wheel(radius={self.radius}m, material={self.material!r}, mass={self.mass})"
