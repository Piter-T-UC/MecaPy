"""Centrifugal clutch design and analysis module.

Units convention: geometry in mm, masses in kg, forces in N, torques in
N*mm and pressures in MPa. Angular speeds are rad/s (with rpm
convenience accessors); radii are converted mm -> m internally where
the m*w^2*r centrifugal force is evaluated.

Reference: standard centrifugal-clutch treatment (shoes flung against a
drum by rotation, retained by springs); Shigley Ch. 16 (Sec. 16-11)
covers the type qualitatively. Engagement is gradual — torque grows
with speed squared above the engagement speed.
"""

import math

from ..base import MechaElement
from .friction_data import get_friction_material
from .friction_interface import DEFAULT_MU


class CentrifugalClutch(MechaElement):
    """
    Centrifugal clutch — spring-retained shoes engage a drum with speed.

    Each shoe of mass m at center-of-gravity radius r_g presses on the
    drum with the net force m*w^2*r_g - S, where S is the spring
    retaining force; below the engagement speed the shoes do not touch
    the drum and no torque is carried.

    Attributes:
        drum_radius (float): Inner drum radius in mm.
        shoe_mass (float): Mass of one shoe in kg.
        cg_radius (float): Radius to the shoe center of gravity in mm.
        n_shoes (int): Number of shoes.
        spring_force (float): Spring retaining force per shoe in N.
        mu (float): Friction coefficient between shoe and drum.
        lining (str): Friction-lining name, or None.
    """

    def __init__(self, drum_radius, shoe_mass, cg_radius, n_shoes,
                 spring_force=0.0, mu=None, lining=None, material="steel",
                 name=None):
        """
        Initialize a CentrifugalClutch object.

        Args:
            drum_radius (float): Inner drum radius in mm.
            shoe_mass (float): Mass of one shoe in kg.
            cg_radius (float): Radius to the shoe center of gravity in mm.
                Must not exceed ``drum_radius``.
            n_shoes (int): Number of shoes (>= 2 for balance).
            spring_force (float): Spring retaining force per shoe in N
                (default 0: shoes touch the drum at any speed).
            mu (float): Friction coefficient. Defaults to the lining's dry
                value when ``lining`` is given, else to the module default.
            lining (str): Optional friction-lining name from
                :mod:`mecapy.clutches.friction_data`.
            material (str): Shoe material type (default: "steel").
            name (str): Optional identifier for the clutch.

        Raises:
            ValueError: For non-positive dimensions or mass, ``cg_radius``
                exceeding ``drum_radius``, ``n_shoes`` below 2, a negative
                spring force, a friction coefficient outside (0, 1.5), or
                an unknown lining name.
        """
        super().__init__(name=name, material=material)
        if drum_radius <= 0:
            raise ValueError("Drum radius must be strictly positive")
        if shoe_mass <= 0:
            raise ValueError("Shoe mass must be strictly positive")
        if cg_radius <= 0:
            raise ValueError("Center-of-gravity radius must be strictly positive")
        if cg_radius > drum_radius:
            raise ValueError("Center-of-gravity radius cannot exceed the drum radius")
        if not isinstance(n_shoes, int) or n_shoes < 2:
            raise ValueError("n_shoes must be an integer >= 2")
        if spring_force < 0:
            raise ValueError("Spring force must be non-negative")

        self._lining_data = None
        if lining is not None:
            self._lining_data = get_friction_material(lining)
        if mu is None:
            mu = self._lining_data["mu_dry"] if self._lining_data else DEFAULT_MU
        if not 0 < mu < 1.5:
            raise ValueError("Friction coefficient must be in (0, 1.5)")

        self.drum_radius = drum_radius
        self.shoe_mass = shoe_mass
        self.cg_radius = cg_radius
        self.n_shoes = n_shoes
        self.spring_force = spring_force
        self.mu = mu
        self.lining = lining

    # ---- Engagement ----

    @property
    def engagement_speed(self):
        """float: Speed w_e = sqrt(S/(m*r_g)) in rad/s at which shoes touch (0 if no spring)."""
        if self.spring_force == 0:
            return 0.0
        r_g = self.cg_radius / 1e3
        return math.sqrt(self.spring_force / (self.shoe_mass * r_g))

    @property
    def engagement_speed_rpm(self):
        """float: Engagement speed in rev/min."""
        return self.engagement_speed * 60 / (2 * math.pi)

    def shoe_normal_force(self, omega):
        """
        Net normal force of one shoe on the drum at a given speed.

        ::

            N = m * w^2 * r_g - S      (0 below the engagement speed)

        Args:
            omega (float): Shaft speed in rad/s.

        Returns:
            float: Normal force per shoe in N (0 when not engaged).

        Raises:
            ValueError: If ``omega`` is negative.
        """
        if omega < 0:
            raise ValueError("Speed must be non-negative")
        r_g = self.cg_radius / 1e3
        force = self.shoe_mass * omega ** 2 * r_g - self.spring_force
        return max(force, 0.0)

    # ---- Torque ----

    def torque(self, omega):
        """
        Torque capacity at a given speed.

        ::

            T = n_shoes * mu * N * r_drum

        Args:
            omega (float): Shaft speed in rad/s.

        Returns:
            float: Torque capacity in N*mm (0 below the engagement speed).

        Raises:
            ValueError: If ``omega`` is negative.
        """
        return (self.n_shoes * self.mu * self.shoe_normal_force(omega)
                * self.drum_radius)

    def speed_for_torque(self, torque):
        """
        Speed at which the clutch carries a given torque.

        Inverts :meth:`torque`.

        Args:
            torque (float): Required torque in N*mm.

        Returns:
            float: Shaft speed in rad/s.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        if torque <= 0:
            raise ValueError("Torque must be strictly positive")
        normal_force = torque / (self.n_shoes * self.mu * self.drum_radius)
        r_g = self.cg_radius / 1e3
        return math.sqrt((normal_force + self.spring_force)
                         / (self.shoe_mass * r_g))

    def contact_pressure(self, omega, shoe_area):
        """
        Shoe/drum contact pressure at a given speed.

        Args:
            omega (float): Shaft speed in rad/s.
            shoe_area (float): Contact area of one shoe in mm^2.

        Returns:
            float: Contact pressure in MPa.

        Raises:
            ValueError: If ``shoe_area`` is not strictly positive or
                ``omega`` is negative.
        """
        if shoe_area <= 0:
            raise ValueError("Shoe area must be strictly positive")
        return self.shoe_normal_force(omega) / shoe_area
