"""Caliper disc brake design and analysis module.

Units convention: geometry in mm, forces in N, torques in N*mm and
pressures in MPa. The pad angle is given in degrees and converted
internally.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 16
(Sec. 16-6, disc brakes, Eqs. 16-34 to 16-40 region). A disc brake
with a FULL annular lining is just a disc clutch used to stop — use
:class:`mecapy.clutches.DiscClutch` (re-exported here as ``DiscBrake``)
for that case; this class models the annular-SECTOR pads of a caliper.
"""

import math

from ..base import MechaElement
from ..clutches.friction_data import get_friction_material
from ..clutches.friction_interface import DEFAULT_MU


class CaliperDiscBrake(MechaElement):
    """
    Caliper disc brake — annular-sector pads clamped on a rotating disc.

    Each pad is a sector of an annulus spanning ``pad_angle`` between
    the inner and outer radii. Both the uniform-pressure (new pads) and
    uniform-wear (worn-in pads, pressure peaks at the inner radius)
    theories are provided as method pairs; each theory reduces to an
    actuating force and an effective friction radius r_e such that
    ``T = n_pads * mu * F * r_e``.

    Attributes:
        outer_radius (float): Pad outer radius ro in mm.
        inner_radius (float): Pad inner radius ri in mm.
        pad_angle (float): Pad angular span in degrees.
        mu (float): Friction coefficient between pad and disc.
        n_pads (int): Number of pads (2 for an opposed caliper).
        lining (str): Friction-lining name, or None.
    """

    def __init__(self, outer_radius, inner_radius, pad_angle, mu=None,
                 n_pads=2, lining=None, material="steel", name=None):
        """
        Initialize a CaliperDiscBrake object.

        Args:
            outer_radius (float): Pad outer radius ro in mm.
            inner_radius (float): Pad inner radius ri in mm. Must be
                smaller than ``outer_radius``.
            pad_angle (float): Pad angular span in degrees, in (0, 360).
            mu (float): Friction coefficient. Defaults to the lining's dry
                value when ``lining`` is given, else to the clutch-module
                default.
            n_pads (int): Number of pads sharing the actuating force
                (default 2, an opposed caliper).
            lining (str): Optional friction-lining name from
                :mod:`mecapy.clutches.friction_data`.
            material (str): Pad backing material type (default: "steel").
            name (str): Optional identifier for the brake.

        Raises:
            ValueError: For non-positive or inverted radii, a pad angle
                outside (0, 360), ``n_pads`` below 1, a friction
                coefficient outside (0, 1.5), or an unknown lining name.
        """
        super().__init__(name=name, material=material)
        if outer_radius <= 0:
            raise ValueError("Outer radius must be strictly positive")
        if inner_radius <= 0:
            raise ValueError("Inner radius must be strictly positive")
        if inner_radius >= outer_radius:
            raise ValueError("Inner radius must be smaller than outer radius")
        if not 0 < pad_angle < 360:
            raise ValueError("Pad angle must be in (0, 360) degrees")
        if not isinstance(n_pads, int) or n_pads < 1:
            raise ValueError("n_pads must be an integer >= 1")

        self._lining_data = None
        if lining is not None:
            self._lining_data = get_friction_material(lining)
        if mu is None:
            mu = self._lining_data["mu_dry"] if self._lining_data else DEFAULT_MU
        if not 0 < mu < 1.5:
            raise ValueError("Friction coefficient must be in (0, 1.5)")

        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.pad_angle = pad_angle
        self.mu = mu
        self.n_pads = n_pads
        self.lining = lining

    # ---- Geometry ----

    @property
    def pad_area(self):
        """float: Area of one pad theta_p*(ro^2 - ri^2)/2 in mm^2."""
        theta_p = math.radians(self.pad_angle)
        return theta_p * (self.outer_radius ** 2 - self.inner_radius ** 2) / 2

    @property
    def effective_radius_uniform_wear(self):
        """float: Effective friction radius r_e = (ro + ri)/2 in mm, uniform wear."""
        return (self.outer_radius + self.inner_radius) / 2

    @property
    def effective_radius_uniform_pressure(self):
        """float: Effective friction radius r_e = (2/3)*(ro^3 - ri^3)/(ro^2 - ri^2) in mm."""
        r3 = self.outer_radius ** 3 - self.inner_radius ** 3
        r2 = self.outer_radius ** 2 - self.inner_radius ** 2
        return 2 * r3 / (3 * r2)

    # ---- Uniform wear (worn-in pads, pressure peaks at ri) ----

    def actuating_force_uniform_wear(self, p_max):
        """
        Actuating force per pad for a given peak pressure, uniform wear.

        ::

            F = theta_p * p_max * ri * (ro - ri)

        Args:
            p_max (float): Peak pad pressure (at the inner radius) in MPa.

        Returns:
            float: Actuating force per pad in N.

        Raises:
            ValueError: If ``p_max`` is not strictly positive.
        """
        if p_max <= 0:
            raise ValueError("p_max must be strictly positive")
        theta_p = math.radians(self.pad_angle)
        return (theta_p * p_max * self.inner_radius
                * (self.outer_radius - self.inner_radius))

    def torque_uniform_wear(self, actuating_force):
        """
        Braking torque for a given per-pad actuating force, uniform wear.

        ::

            T = n_pads * mu * F * (ro + ri) / 2

        Args:
            actuating_force (float): Actuating force per pad in N.

        Returns:
            float: Braking torque in N*mm.

        Raises:
            ValueError: If ``actuating_force`` is not strictly positive.
        """
        self._check_force(actuating_force)
        return (self.n_pads * self.mu * actuating_force
                * self.effective_radius_uniform_wear)

    def max_pressure_uniform_wear(self, actuating_force):
        """
        Peak pad pressure for a given per-pad actuating force, uniform wear.

        Inverts :meth:`actuating_force_uniform_wear`; the peak occurs at
        the inner radius.

        Args:
            actuating_force (float): Actuating force per pad in N.

        Returns:
            float: Peak pressure in MPa.

        Raises:
            ValueError: If ``actuating_force`` is not strictly positive.
        """
        self._check_force(actuating_force)
        theta_p = math.radians(self.pad_angle)
        return (actuating_force
                / (theta_p * self.inner_radius
                   * (self.outer_radius - self.inner_radius)))

    # ---- Uniform pressure (new pads) ----

    def actuating_force_uniform_pressure(self, pressure):
        """
        Actuating force per pad for a given uniform pressure.

        ::

            F = theta_p * p * (ro^2 - ri^2) / 2

        Args:
            pressure (float): Uniform pad pressure in MPa.

        Returns:
            float: Actuating force per pad in N.

        Raises:
            ValueError: If ``pressure`` is not strictly positive.
        """
        if pressure <= 0:
            raise ValueError("Pressure must be strictly positive")
        return pressure * self.pad_area

    def torque_uniform_pressure(self, actuating_force):
        """
        Braking torque for a given per-pad actuating force, uniform pressure.

        ::

            T = n_pads * mu * F * (2/3) * (ro^3 - ri^3) / (ro^2 - ri^2)

        Args:
            actuating_force (float): Actuating force per pad in N.

        Returns:
            float: Braking torque in N*mm.

        Raises:
            ValueError: If ``actuating_force`` is not strictly positive.
        """
        self._check_force(actuating_force)
        return (self.n_pads * self.mu * actuating_force
                * self.effective_radius_uniform_pressure)

    def pressure_uniform_pressure(self, actuating_force):
        """
        Uniform pad pressure for a given per-pad actuating force.

        Inverts :meth:`actuating_force_uniform_pressure`.

        Args:
            actuating_force (float): Actuating force per pad in N.

        Returns:
            float: Uniform pressure in MPa.

        Raises:
            ValueError: If ``actuating_force`` is not strictly positive.
        """
        self._check_force(actuating_force)
        return actuating_force / self.pad_area

    # ---- Lining check ----

    def pressure_safety_factor(self, actuating_force, theory="uniform_wear",
                               p_allowable=None):
        """
        Ratio of the lining's allowable pressure to the working pressure.

        Args:
            actuating_force (float): Actuating force per pad in N.
            theory (str): "uniform_wear" (default, conservative) or
                "uniform_pressure".
            p_allowable (float): Allowable pressure in MPa. Defaults to
                the lining's ``p_max`` when a lining was given.

        Returns:
            float: p_allowable / p_working.

        Raises:
            ValueError: If no allowable pressure is available (no lining
                and no ``p_allowable``), or ``theory`` is unknown.
        """
        if p_allowable is None:
            if self._lining_data is None:
                raise ValueError(
                    "No allowable pressure: give p_allowable or construct "
                    "the brake with a lining"
                )
            p_allowable = self._lining_data["p_max"]
        if theory == "uniform_wear":
            p_working = self.max_pressure_uniform_wear(actuating_force)
        elif theory == "uniform_pressure":
            p_working = self.pressure_uniform_pressure(actuating_force)
        else:
            raise ValueError("theory must be 'uniform_wear' or 'uniform_pressure'")
        return p_allowable / p_working

    # ---- Internal helpers ----

    @staticmethod
    def _check_force(actuating_force):
        """Validate a per-pad actuating force."""
        if actuating_force <= 0:
            raise ValueError("Actuating force must be strictly positive")
