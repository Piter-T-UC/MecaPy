"""Flat-belt design and analysis module.

Units convention: geometry in mm, forces in N, power in W, mass per length
in kg/m and stresses in MPa. Belt/pulley speed is in m/s.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 17
(Sec. 17-2, flat- and round-belt drives, Eqs. 17-1 to 17-13).
"""

import math

from ..base import MechaElement
from ..clutches.friction_interface import DEFAULT_MU
from .belt_data import get_flat_belt_material


class FlatBelt(MechaElement):
    """
    Flat belt wrapped open or crossed around two pulleys.

    The belt develops the capstan tension ratio
    ``F1/F2 = exp(mu*phi)`` between the tight and slack sides over the
    smaller (governing) wrap angle; centrifugal tension Fc reduces the
    usable friction because only (F - Fc) presses the belt onto the
    pulley. :class:`~mecapy.belts.vbelt.VBelt` reuses this whole capstan
    core and overrides only the groove-wedging friction.

    Attributes:
        width (float): Belt width b in mm.
        thickness (float): Belt thickness t in mm.
        driver_diameter (float): Driver pulley diameter in mm.
        driven_diameter (float): Driven pulley diameter in mm.
        center_distance (float): Center-to-center distance C in mm.
        mu (float): Friction coefficient in use.
        density (float): Belt material density in kg/m^3, or None.
        allowable_stress (float): Allowable working stress in MPa, or None.
        drive (str): "open" or "crossed".
        belt_material (str): Flat-belt material name, or None.
    """

    def __init__(self, width, thickness, driver_diameter, driven_diameter,
                 center_distance, mu=None, belt_material=None, density=None,
                 allowable_stress=None, drive="open", material="steel",
                 name=None):
        """
        Initialize a FlatBelt object.

        Args:
            width (float): Belt width b in mm.
            thickness (float): Belt thickness t in mm.
            driver_diameter (float): Driver pulley diameter in mm.
            driven_diameter (float): Driven pulley diameter in mm.
            center_distance (float): Center-to-center distance C in mm.
                Must satisfy the open/crossed geometric minimum.
            mu (float): Friction coefficient. Defaults to the material's
                value when ``belt_material`` is given, else to the
                clutch-module default. An explicit value always wins.
            belt_material (str): Optional flat-belt material name from
                :mod:`mecapy.belts.belt_data`; supplies default ``mu``,
                ``density`` and ``allowable_stress``.
            density (float): Belt material density in kg/m^3. Defaults to
                the material's value; stays None with no material given,
                and dependent methods then raise.
            allowable_stress (float): Allowable working stress in MPa.
                Same defaulting/None behavior as ``density``.
            drive (str): "open" (default) or "crossed".
            material (str): Structural material type, for the inherited
                stress/safety calculations (default: "steel").
            name (str): Optional identifier for the belt.

        Raises:
            ValueError: For non-positive dimensions, an unknown ``drive``,
                a friction coefficient outside (0, 1.5), a center distance
                too small for the chosen drive geometry, or an unknown
                ``belt_material``.
        """
        super().__init__(name=name, material=material)
        if width <= 0:
            raise ValueError("Belt width must be strictly positive")
        if thickness <= 0:
            raise ValueError("Belt thickness must be strictly positive")
        if driver_diameter <= 0:
            raise ValueError("Driver diameter must be strictly positive")
        if driven_diameter <= 0:
            raise ValueError("Driven diameter must be strictly positive")
        if center_distance <= 0:
            raise ValueError("Center distance must be strictly positive")
        if drive not in ("open", "crossed"):
            raise ValueError("drive must be 'open' or 'crossed'")
        if drive == "open":
            if center_distance <= abs(driven_diameter - driver_diameter) / 2:
                raise ValueError("Center distance too small for an open-belt drive")
        else:
            if center_distance <= (driven_diameter + driver_diameter) / 2:
                raise ValueError("Center distance too small for a crossed-belt drive")

        self._material_data = None
        if belt_material is not None:
            self._material_data = get_flat_belt_material(belt_material)
        if mu is None:
            mu = self._material_data["mu"] if self._material_data else DEFAULT_MU
        if not 0 < mu < 1.5:
            raise ValueError("Friction coefficient must be in (0, 1.5)")
        if density is None and self._material_data:
            density = self._material_data["density"]
        if allowable_stress is None and self._material_data:
            allowable_stress = self._material_data["allowable_stress"]

        self.width = width
        self.thickness = thickness
        self.driver_diameter = driver_diameter
        self.driven_diameter = driven_diameter
        self.center_distance = center_distance
        self.mu = mu
        self.density = density
        self.allowable_stress = allowable_stress
        self.drive = drive
        self.belt_material = belt_material

    # ---- Geometry ----

    @property
    def speed_ratio(self):
        """float: Driven/driver speed ratio = driver_diameter / driven_diameter."""
        return self.driver_diameter / self.driven_diameter

    @property
    def _half_angle_arg(self):
        """float: sin of the half-angle subtended at the center by the two tangent points."""
        if self.drive == "open":
            return abs(self.driven_diameter - self.driver_diameter) / (2 * self.center_distance)
        return (self.driven_diameter + self.driver_diameter) / (2 * self.center_distance)

    @property
    def wrap_angle_driver(self):
        """float: Belt wrap angle at the driver pulley, in degrees (Eq. 17-1)."""
        delta = math.degrees(2 * math.asin(self._half_angle_arg))
        if self.drive == "crossed":
            return 180.0 + delta
        return 180.0 - delta if self.driver_diameter <= self.driven_diameter else 180.0 + delta

    @property
    def wrap_angle_driven(self):
        """float: Belt wrap angle at the driven pulley, in degrees (Eq. 17-1)."""
        delta = math.degrees(2 * math.asin(self._half_angle_arg))
        if self.drive == "crossed":
            return 180.0 + delta
        return 180.0 - delta if self.driven_diameter <= self.driver_diameter else 180.0 + delta

    @property
    def governing_wrap_angle(self):
        """float: Smaller of the two wrap angles, in degrees (slip governs there)."""
        return min(self.wrap_angle_driver, self.wrap_angle_driven)

    @property
    def belt_length(self):
        """float: Belt (pitch) length in mm (Eq. 17-2 open / Eq. 17-3 crossed)."""
        driven_d, driver_d = self.driven_diameter, self.driver_diameter
        phi_driver = math.radians(self.wrap_angle_driver)
        phi_driven = math.radians(self.wrap_angle_driven)
        if self.drive == "open":
            term1 = math.sqrt(4 * self.center_distance ** 2 - (driven_d - driver_d) ** 2)
        else:
            term1 = math.sqrt(4 * self.center_distance ** 2 - (driven_d + driver_d) ** 2)
        term2 = (phi_driven * driven_d + phi_driver * driver_d) / 2
        return term1 + term2

    # ---- Friction and tensions ----

    @property
    def effective_friction(self):
        """float: Friction coefficient used in the capstan equation (override hook for VBelt)."""
        return self.mu

    @property
    def tension_ratio(self):
        """float: Tight/slack tension ratio F1/F2 = exp(mu*phi) over the governing wrap angle."""
        return math.exp(self.effective_friction * math.radians(self.governing_wrap_angle))

    @property
    def mass_per_length(self):
        """float: Belt mass per unit length in kg/m = density * width * thickness.

        Raises:
            ValueError: If ``density`` was not given (directly or via
                ``belt_material``).
        """
        if self.density is None:
            raise ValueError("density is required to compute mass_per_length")
        return self.density * self.width * self.thickness * 1e-6

    @property
    def allowable_tension(self):
        """float: Allowable tight-side tension in N = allowable_stress * width * thickness.

        Raises:
            ValueError: If ``allowable_stress`` was not given (directly or
                via ``belt_material``).
        """
        if self.allowable_stress is None:
            raise ValueError("allowable_stress is required to compute allowable_tension")
        return self.allowable_stress * self.width * self.thickness

    def belt_speed(self, rpm):
        """
        Belt (pitch-line) speed at the driver pulley.

        Args:
            rpm (float): Driver pulley speed in rev/min.

        Returns:
            float: Belt speed in m/s.

        Raises:
            ValueError: If ``rpm`` is not strictly positive.
        """
        if rpm <= 0:
            raise ValueError("Speed must be strictly positive")
        return math.pi * self.driver_diameter * rpm / 60000.0

    def centrifugal_tension(self, rpm):
        """
        Centrifugal tension Fc = m' * V^2 at a given driver speed.

        Args:
            rpm (float): Driver pulley speed in rev/min.

        Returns:
            float: Centrifugal tension in N.

        Raises:
            ValueError: See :meth:`belt_speed`; or if ``density`` is
                unavailable (see :attr:`mass_per_length`).
        """
        v = self.belt_speed(rpm)
        return self.mass_per_length * v ** 2

    def tensions_for_power(self, power, rpm):
        """
        Tight- and slack-side tensions for a given transmitted power, at
        full capstan development (incipient slip -- the minimum tensions
        that can deliver this power).

        Args:
            power (float): Transmitted power in W.
            rpm (float): Driver pulley speed in rev/min.

        Returns:
            tuple: (F1, F2), tight- and slack-side tensions in N.

        Raises:
            ValueError: If ``power`` is not strictly positive; see
                :meth:`belt_speed` and :attr:`mass_per_length`.
        """
        if power <= 0:
            raise ValueError("Power must be strictly positive")
        v = self.belt_speed(rpm)
        delta_f = power / v
        fc = self.centrifugal_tension(rpm)
        ratio = self.tension_ratio
        f1 = ((fc + delta_f) * ratio - fc) / (ratio - 1)
        f2 = f1 - delta_f
        return f1, f2

    def power(self, tight_tension, rpm):
        """
        Transmitted power for a given tight-side tension, at full capstan
        development. Exact inverse of :meth:`tensions_for_power`.

        Args:
            tight_tension (float): Tight-side tension F1 in N.
            rpm (float): Driver pulley speed in rev/min.

        Returns:
            float: Transmitted power in W.

        Raises:
            ValueError: If ``tight_tension`` does not exceed the
                centrifugal tension, or is not strictly positive.
        """
        if tight_tension <= 0:
            raise ValueError("Tight tension must be strictly positive")
        fc = self.centrifugal_tension(rpm)
        if tight_tension <= fc:
            raise ValueError("Tight tension must exceed the centrifugal tension")
        ratio = self.tension_ratio
        f2 = fc + (tight_tension - fc) / ratio
        delta_f = tight_tension - f2
        v = self.belt_speed(rpm)
        return delta_f * v

    def initial_tension(self, power, rpm):
        """
        Initial (installation) tension for a given transmitted power.

        ::

            Fi = (F1 + F2) / 2 - Fc                (Eq. 17-9)

        Args:
            power (float): Transmitted power in W.
            rpm (float): Driver pulley speed in rev/min.

        Returns:
            float: Initial tension in N.

        Raises:
            ValueError: See :meth:`tensions_for_power`.
        """
        f1, f2 = self.tensions_for_power(power, rpm)
        fc = self.centrifugal_tension(rpm)
        return (f1 + f2) / 2 - fc

    def power_capacity(self, rpm):
        """
        Power capacity at the allowable tension, at a given speed.

        Args:
            rpm (float): Driver pulley speed in rev/min.

        Returns:
            float: Power capacity in W.

        Raises:
            ValueError: See :attr:`allowable_tension` and :meth:`power`.
        """
        return self.power(self.allowable_tension, rpm)

    def power_safety_factor(self, power, rpm):
        """
        Ratio of the power capacity to a given transmitted power.

        Args:
            power (float): Transmitted power in W.
            rpm (float): Driver pulley speed in rev/min.

        Returns:
            float: power_capacity(rpm) / power.

        Raises:
            ValueError: If ``power`` is not strictly positive; see
                :meth:`power_capacity`.
        """
        if power <= 0:
            raise ValueError("Power must be strictly positive")
        return self.power_capacity(rpm) / power

    # ---- Belt stress ----

    def belt_stress(self, tight_tension):
        """
        Tensile stress in the belt at the tight side.

        Args:
            tight_tension (float): Tight-side tension F1 in N.

        Returns:
            float: Belt tensile stress in MPa.

        Raises:
            ValueError: If ``tight_tension`` is not strictly positive.
        """
        if tight_tension <= 0:
            raise ValueError("Tight tension must be strictly positive")
        return tight_tension / (self.width * self.thickness)

    def belt_stress_safety_factor(self, tight_tension):
        """
        Safety factor of the belt against yielding, at the tight side.

        Args:
            tight_tension (float): Tight-side tension F1 in N.

        Returns:
            float: Ratio of yield strength to the belt tensile stress.

        Raises:
            ValueError: See :meth:`belt_stress`.
        """
        stress = self.belt_stress(tight_tension)
        sy = self.material_properties["yield_strength"] / 1e6
        return sy / stress
