"""Band brake design and analysis module.

Units convention: geometry in mm, forces in N, torques in N*mm and
pressures/stresses in MPa. The wrap angle is given in degrees and
converted internally.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 16
(Sec. 16-5 region, band-type clutches and brakes, Eqs. 16-14 to 16-17).
"""

import math

from ..base import MechaElement
from ..clutches.friction_data import get_friction_material
from ..clutches.friction_interface import DEFAULT_MU


class BandBrake(MechaElement):
    """
    Band brake — a flexible band wrapped around a rotating drum.

    Tightening the band develops the belt-friction tension ratio
    ``P1/P2 = exp(mu*phi)`` between the tight and slack ends; the
    difference in tensions times the drum radius is the braking torque.
    The peak lining pressure occurs at the tight end.

    Attributes:
        drum_diameter (float): Drum diameter D in mm.
        band_width (float): Band (lining) width b in mm.
        wrap_angle (float): Angle of wrap phi in degrees.
        mu (float): Friction coefficient between band and drum.
        band_thickness (float): Band metal thickness t in mm, or None.
        lining (str): Friction-lining name, or None.
    """

    def __init__(self, drum_diameter, band_width, wrap_angle, mu=None,
                 lining=None, band_thickness=None, material="steel", name=None):
        """
        Initialize a BandBrake object.

        Args:
            drum_diameter (float): Drum diameter D in mm.
            band_width (float): Band width b in mm.
            wrap_angle (float): Angle of wrap phi in degrees, in (0, 1080]
                (up to three full turns).
            mu (float): Friction coefficient. Defaults to the lining's dry
                value when ``lining`` is given, else to the clutch-module
                default.
            lining (str): Optional friction-lining name from
                :mod:`mecapy.clutches.friction_data`.
            band_thickness (float): Optional band metal thickness t in mm,
                needed for :meth:`band_stress`.
            material (str): Band material type (default: "steel").
            name (str): Optional identifier for the brake.

        Raises:
            ValueError: For non-positive dimensions, a wrap angle outside
                (0, 1080] degrees, a friction coefficient outside
                (0, 1.5), or an unknown lining name.
        """
        super().__init__(name=name, material=material)
        if drum_diameter <= 0:
            raise ValueError("Drum diameter must be strictly positive")
        if band_width <= 0:
            raise ValueError("Band width must be strictly positive")
        if not 0 < wrap_angle <= 1080:
            raise ValueError("Wrap angle must be in (0, 1080] degrees")
        if band_thickness is not None and band_thickness <= 0:
            raise ValueError("Band thickness must be strictly positive")

        self._lining_data = None
        if lining is not None:
            self._lining_data = get_friction_material(lining)
        if mu is None:
            mu = self._lining_data["mu_dry"] if self._lining_data else DEFAULT_MU
        if not 0 < mu < 1.5:
            raise ValueError("Friction coefficient must be in (0, 1.5)")

        self.drum_diameter = drum_diameter
        self.band_width = band_width
        self.wrap_angle = wrap_angle
        self.mu = mu
        self.band_thickness = band_thickness
        self.lining = lining

    # ---- Tensions and torque ----

    @property
    def tension_ratio(self):
        """float: Tight/slack tension ratio P1/P2 = exp(mu*phi) (Eq. 16-14)."""
        return math.exp(self.mu * math.radians(self.wrap_angle))

    def slack_tension(self, tight_tension):
        """
        Slack-side tension for a given tight-side tension.

        Args:
            tight_tension (float): Tight-side tension P1 in N.

        Returns:
            float: Slack-side tension P2 = P1 / exp(mu*phi) in N.

        Raises:
            ValueError: If ``tight_tension`` is not strictly positive.
        """
        self._check_tension(tight_tension)
        return tight_tension / self.tension_ratio

    def torque(self, tight_tension):
        """
        Braking torque for a given tight-side tension.

        ::

            T = (P1 - P2) * D / 2                  (Eq. 16-15)

        Args:
            tight_tension (float): Tight-side tension P1 in N.

        Returns:
            float: Braking torque in N*mm.

        Raises:
            ValueError: If ``tight_tension`` is not strictly positive.
        """
        self._check_tension(tight_tension)
        return ((tight_tension - self.slack_tension(tight_tension))
                * self.drum_diameter / 2)

    def tight_tension_for_torque(self, torque):
        """
        Tight-side tension needed to develop a given torque.

        Inverts :meth:`torque`.

        Args:
            torque (float): Required braking torque in N*mm.

        Returns:
            float: Tight-side tension P1 in N.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        if torque <= 0:
            raise ValueError("Torque must be strictly positive")
        return (2 * torque
                / (self.drum_diameter * (1 - 1 / self.tension_ratio)))

    # ---- Pressure and band stress ----

    def max_pressure(self, tight_tension):
        """
        Peak lining pressure, at the tight end of the band.

        ::

            p = 2 * P1 / (b * D)                   (Eq. 16-17)

        Args:
            tight_tension (float): Tight-side tension P1 in N.

        Returns:
            float: Peak pressure in MPa.

        Raises:
            ValueError: If ``tight_tension`` is not strictly positive.
        """
        self._check_tension(tight_tension)
        return 2 * tight_tension / (self.band_width * self.drum_diameter)

    def tight_tension_for_pressure(self, p_max):
        """
        Tight-side tension that produces a given peak lining pressure.

        Inverts Eq. 16-17 — the design entry point when the lining's
        pressure rating governs.

        Args:
            p_max (float): Allowable peak pressure in MPa.

        Returns:
            float: Tight-side tension P1 in N.

        Raises:
            ValueError: If ``p_max`` is not strictly positive.
        """
        if p_max <= 0:
            raise ValueError("p_max must be strictly positive")
        return p_max * self.band_width * self.drum_diameter / 2

    def pressure_safety_factor(self, tight_tension, p_allowable=None):
        """
        Ratio of the lining's allowable pressure to the peak pressure.

        Args:
            tight_tension (float): Tight-side tension P1 in N.
            p_allowable (float): Allowable pressure in MPa. Defaults to
                the lining's ``p_max`` when a lining was given.

        Returns:
            float: p_allowable / p_peak.

        Raises:
            ValueError: If no allowable pressure is available (no lining
                and no ``p_allowable``).
        """
        if p_allowable is None:
            if self._lining_data is None:
                raise ValueError(
                    "No allowable pressure: give p_allowable or construct "
                    "the brake with a lining"
                )
            p_allowable = self._lining_data["p_max"]
        return p_allowable / self.max_pressure(tight_tension)

    def band_stress(self, tight_tension):
        """
        Tensile stress in the band metal at the tight end.

        ::

            sigma = P1 / (b * t)

        Args:
            tight_tension (float): Tight-side tension P1 in N.

        Returns:
            float: Band tensile stress in MPa.

        Raises:
            ValueError: If ``band_thickness`` was not given, or
                ``tight_tension`` is not strictly positive.
        """
        self._check_tension(tight_tension)
        if self.band_thickness is None:
            raise ValueError("band_thickness is required to compute the band stress")
        return tight_tension / (self.band_width * self.band_thickness)

    def band_safety_factor(self, tight_tension):
        """
        Safety factor of the band metal against yielding.

        Uses the material yield strength (converted Pa -> MPa), like
        :meth:`mecapy.shafts.PowerScrew.screw_safety_factor`.

        Args:
            tight_tension (float): Tight-side tension P1 in N.

        Returns:
            float: Ratio of yield strength to the band tensile stress.

        Raises:
            ValueError: See :meth:`band_stress`.
        """
        stress = self.band_stress(tight_tension)
        sy = self.material_properties["yield_strength"] / 1e6
        return sy / stress

    # ---- Internal helpers ----

    @staticmethod
    def _check_tension(tight_tension):
        """Validate a tight-side tension."""
        if tight_tension <= 0:
            raise ValueError("Tight-side tension must be strictly positive")
