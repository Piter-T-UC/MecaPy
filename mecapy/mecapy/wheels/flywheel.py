"""Flywheel design and analysis module.

Units convention: this module is SI throughout — geometry in m, mass in
kg, angular velocity in rad/s, energy in J, inertia in kg*m^2 and
stresses in Pa — because it extends the SI-based :class:`Wheel` and
consumes the material database (density kg/m^3, strengths Pa) directly.
This deliberately differs from the mm/MPa convention of the newer
geometry modules; convert at the boundary if combining the two.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 16
(Sec. 16-12, flywheels: energy fluctuation and the coefficient of
speed fluctuation) and Ch. 3 (Eq. 3-55, rotating-ring/disc stresses).
"""

import math

from .wheel import Wheel


class Flywheel(Wheel):
    """
    Flywheel — an annular (or solid) disc storing rotational energy.

    Covers the two design questions of Shigley Sec. 16-12: sizing the
    inertia to hold the speed fluctuation of a torque cycle within a
    chosen coefficient of fluctuation, and checking the rotating-disc
    stresses that limit the maximum speed.

    Attributes:
        outer_radius (float): Outer radius in m (alias of ``radius``).
        inner_radius (float): Bore radius in m (0 for a solid disc).
        thickness (float): Axial thickness in m, or None when the mass
            was given directly.
        mass (float): Mass in kg.
    """

    def __init__(self, outer_radius, inner_radius=0.0, thickness=None,
                 mass=None, material="steel", name=None):
        """
        Initialize a Flywheel object.

        Args:
            outer_radius (float): Outer radius in m.
            inner_radius (float): Bore radius in m (default 0, solid
                disc). Must be smaller than ``outer_radius``.
            thickness (float): Axial thickness in m. Give exactly one of
                ``thickness`` (mass computed from the geometry and the
                material density) or ``mass``.
            mass (float): Flywheel mass in kg.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the flywheel.

        Raises:
            ValueError: For non-positive dimensions, an inner radius not
                below the outer radius, or unless exactly one of
                ``thickness`` / ``mass`` is given.
        """
        if outer_radius <= 0:
            raise ValueError("Outer radius must be strictly positive")
        if inner_radius < 0:
            raise ValueError("Inner radius must be non-negative")
        if inner_radius >= outer_radius:
            raise ValueError("Inner radius must be smaller than the outer radius")
        if (thickness is None) == (mass is None):
            raise ValueError("Give exactly one of thickness or mass")
        if thickness is not None and thickness <= 0:
            raise ValueError("Thickness must be strictly positive")
        if mass is not None and mass <= 0:
            raise ValueError("Mass must be strictly positive")

        super().__init__(radius=outer_radius, material=material, mass=mass,
                         name=name)
        self.inner_radius = inner_radius
        self.thickness = thickness
        if mass is None:
            density = self.material_properties["density"]
            self.mass = (density * math.pi
                         * (outer_radius ** 2 - inner_radius ** 2) * thickness)

    @property
    def outer_radius(self):
        """float: Outer radius in m (alias of the inherited ``radius``)."""
        return self.radius

    @property
    def moment_of_inertia(self):
        """float: Moment of inertia of the annular cylinder I = m*(ro^2 + ri^2)/2 in kg*m^2."""
        return 0.5 * self.mass * (self.outer_radius ** 2 + self.inner_radius ** 2)

    # ---- Energy fluctuation and sizing ----

    def energy_fluctuation(self, omega_max, omega_min):
        """
        Energy released between the maximum and minimum cycle speeds.

        ::

            Ue = (1/2) * I * (w_max^2 - w_min^2)

        Args:
            omega_max (float): Maximum cycle speed in rad/s.
            omega_min (float): Minimum cycle speed in rad/s.

        Returns:
            float: Energy fluctuation Ue in J.

        Raises:
            ValueError: If a speed is negative or ``omega_min`` exceeds
                ``omega_max``.
        """
        self._check_speed_pair(omega_max, omega_min)
        return 0.5 * self.moment_of_inertia * (omega_max ** 2 - omega_min ** 2)

    @staticmethod
    def coefficient_of_fluctuation(omega_max, omega_min):
        """
        Coefficient of speed fluctuation of a cycle.

        ::

            Cs = (w_max - w_min) / w_avg,   w_avg = (w_max + w_min)/2

        Args:
            omega_max (float): Maximum cycle speed in rad/s.
            omega_min (float): Minimum cycle speed in rad/s.

        Returns:
            float: Coefficient of fluctuation Cs (dimensionless).

        Raises:
            ValueError: If a speed is negative, ``omega_min`` exceeds
                ``omega_max``, or both speeds are zero.
        """
        Flywheel._check_speed_pair(omega_max, omega_min)
        omega_avg = (omega_max + omega_min) / 2
        if omega_avg == 0:
            raise ValueError("Average speed must be strictly positive")
        return (omega_max - omega_min) / omega_avg

    @staticmethod
    def required_inertia(energy_fluctuation, cs, omega_avg):
        """
        Inertia needed to hold a speed fluctuation to a chosen Cs.

        The sizing entry point — usable before any geometry exists::

            I = Ue / (Cs * w_avg^2)

        Args:
            energy_fluctuation (float): Cycle energy fluctuation Ue in J.
            cs (float): Target coefficient of speed fluctuation.
            omega_avg (float): Average operating speed in rad/s.

        Returns:
            float: Required moment of inertia in kg*m^2.

        Raises:
            ValueError: If any argument is not strictly positive.
        """
        if energy_fluctuation <= 0:
            raise ValueError("Energy fluctuation must be strictly positive")
        if cs <= 0:
            raise ValueError("Coefficient of fluctuation must be strictly positive")
        if omega_avg <= 0:
            raise ValueError("Average speed must be strictly positive")
        return energy_fluctuation / (cs * omega_avg ** 2)

    def speed_swing(self, energy_fluctuation, omega_avg):
        """
        Cycle speed extremes produced by an energy fluctuation.

        Uses ``Ue = I * w_avg * (w_max - w_min)`` (exact for the
        average-speed definition of Cs).

        Args:
            energy_fluctuation (float): Cycle energy fluctuation Ue in J.
            omega_avg (float): Average operating speed in rad/s.

        Returns:
            tuple: ``(omega_max, omega_min)`` in rad/s.

        Raises:
            ValueError: If an argument is not strictly positive, or the
                fluctuation is too large for the flywheel (the minimum
                speed would be negative).
        """
        if energy_fluctuation <= 0:
            raise ValueError("Energy fluctuation must be strictly positive")
        if omega_avg <= 0:
            raise ValueError("Average speed must be strictly positive")
        delta = energy_fluctuation / (self.moment_of_inertia * omega_avg)
        omega_min = omega_avg - delta / 2
        if omega_min < 0:
            raise ValueError(
                "Energy fluctuation too large for this inertia: the minimum "
                "speed would be negative"
            )
        return omega_avg + delta / 2, omega_min

    # ---- Rotating-disc stresses ----

    def tangential_stress(self, omega, radius=None):
        """
        Tangential (hoop) stress of the rotating disc at a radius.

        Full rotating annular-disc solution (Shigley Eq. 3-55)::

            sigma_t = rho*w^2 * (3+nu)/8
                      * (ri^2 + ro^2 + ri^2*ro^2/r^2 - (1+3nu)/(3+nu) * r^2)

        with the solid-disc branch for ``inner_radius = 0``.

        Args:
            omega (float): Angular velocity in rad/s.
            radius (float): Radius r in m at which to evaluate. Defaults
                to the location of the maximum (the bore for an annulus,
                the center for a solid disc).

        Returns:
            float: Tangential stress in Pa.

        Raises:
            ValueError: If ``omega`` is negative or ``radius`` is outside
                the disc.
        """
        rho = self.material_properties["density"]
        nu = self.material_properties["poisson_ratio"]
        ro, ri = self.outer_radius, self.inner_radius
        r = self._stress_radius(radius)
        if omega < 0:
            raise ValueError("Speed must be non-negative")
        common = rho * omega ** 2 * (3 + nu) / 8
        if ri == 0:
            return common * (ro ** 2 - (1 + 3 * nu) / (3 + nu) * r ** 2)
        return common * (ri ** 2 + ro ** 2 + ri ** 2 * ro ** 2 / r ** 2
                         - (1 + 3 * nu) / (3 + nu) * r ** 2)

    def radial_stress(self, omega, radius=None):
        """
        Radial stress of the rotating disc at a radius.

        ::

            sigma_r = rho*w^2 * (3+nu)/8
                      * (ri^2 + ro^2 - ri^2*ro^2/r^2 - r^2)

        Zero at both free surfaces; the solid-disc branch applies for
        ``inner_radius = 0``. Defaults to the geometric mean radius
        sqrt(ri*ro) where the annular maximum occurs (the center for a
        solid disc).

        Args:
            omega (float): Angular velocity in rad/s.
            radius (float): Radius r in m at which to evaluate.

        Returns:
            float: Radial stress in Pa.

        Raises:
            ValueError: If ``omega`` is negative or ``radius`` is outside
                the disc.
        """
        rho = self.material_properties["density"]
        nu = self.material_properties["poisson_ratio"]
        ro, ri = self.outer_radius, self.inner_radius
        if omega < 0:
            raise ValueError("Speed must be non-negative")
        if radius is None:
            radius = math.sqrt(ri * ro) if ri > 0 else 0.0
        else:
            radius = self._stress_radius(radius)
        common = rho * omega ** 2 * (3 + nu) / 8
        if ri == 0:
            return common * (ro ** 2 - radius ** 2)
        return common * (ri ** 2 + ro ** 2 - ri ** 2 * ro ** 2 / radius ** 2
                         - radius ** 2)

    def rim_hoop_stress(self, omega):
        """
        Thin-rotating-ring hoop stress, a quick rim check.

        ::

            sigma = rho * w^2 * r_mean^2

        Args:
            omega (float): Angular velocity in rad/s.

        Returns:
            float: Hoop stress in Pa (approximate; use
            :meth:`tangential_stress` for the full solution).

        Raises:
            ValueError: If ``omega`` is negative.
        """
        if omega < 0:
            raise ValueError("Speed must be non-negative")
        rho = self.material_properties["density"]
        r_mean = (self.outer_radius + self.inner_radius) / 2
        return rho * omega ** 2 * r_mean ** 2

    def burst_safety_factor(self, omega):
        """
        Safety factor against yielding at the peak tangential stress.

        Args:
            omega (float): Angular velocity in rad/s.

        Returns:
            float: Yield strength / maximum tangential stress (both Pa).

        Raises:
            ValueError: If ``omega`` is not strictly positive.
        """
        if omega <= 0:
            raise ValueError("Speed must be strictly positive")
        return (self.material_properties["yield_strength"]
                / self.tangential_stress(omega))

    def max_speed(self, safety_factor=1.0):
        """
        Speed at which the peak tangential stress reaches yield/n.

        The disc stresses scale with ``omega^2``, so::

            w_max = sqrt( Sy / (n * sigma_t(w=1)) )

        Args:
            safety_factor (float): Required safety factor n (default 1,
                the burst-onset speed).

        Returns:
            float: Maximum speed in rad/s.

        Raises:
            ValueError: If ``safety_factor`` is not strictly positive.
        """
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        unit_stress = self.tangential_stress(1.0)
        sy = self.material_properties["yield_strength"]
        return math.sqrt(sy / (safety_factor * unit_stress))

    def max_speed_rpm(self, safety_factor=1.0):
        """
        Maximum speed in rev/min (see :meth:`max_speed`).

        Args:
            safety_factor (float): Required safety factor n (default 1).

        Returns:
            float: Maximum speed in rev/min.
        """
        return self.max_speed(safety_factor) * 60 / (2 * math.pi)

    # ---- Internal helpers ----

    @staticmethod
    def _check_speed_pair(omega_max, omega_min):
        """Validate a (max, min) cycle-speed pair."""
        if omega_max < 0 or omega_min < 0:
            raise ValueError("Speeds must be non-negative")
        if omega_min > omega_max:
            raise ValueError("omega_min must not exceed omega_max")

    def _stress_radius(self, radius):
        """Default and validate the evaluation radius for the disc stresses."""
        ro, ri = self.outer_radius, self.inner_radius
        if radius is None:
            return ri if ri > 0 else 0.0
        if not ri <= radius <= ro:
            raise ValueError("Radius must lie within the disc")
        return radius

    def __repr__(self):
        return (f"Flywheel(outer_radius={self.radius}m, "
                f"inner_radius={self.inner_radius}m, "
                f"material={self.material!r}, mass={self.mass})")
