"""Hydrodynamic journal bearing analysis (Shigley ch. 12).

Units: geometry in mm, journal speed in rev/s, load in N, viscosity in
mPa*s, pressures in MPa, temperatures in degrees Celsius.
"""

import math

from ..base import MechaElement
from .lubrication_data import (
    TRUMPLER_MAX_STARTUP_PRESSURE,
    TRUMPLER_MAX_TEMPERATURE,
    TRUMPLER_MIN_FILM_BASE,
    TRUMPLER_MIN_FILM_SLOPE,
    is_thick_film,
    raimondi_boyd,
    viscosity as sae_viscosity,
)


class JournalBearing(MechaElement):
    """
    Hydrodynamic journal bearing analysis.

    Implements the Shigley ch. 12 design methodology: Petroff friction
    (eq. 12-6), Sommerfeld number (eq. 12-7), the fig. 12-4 thick-film
    stability check, Raimondi-Boyd performance charts (figs. 12-16 to
    12-24), lubricant temperature rise and the Trumpler design criteria
    (sec. 12-14).

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`.

    Attributes:
        radius (float): Journal radius r in mm.
        clearance (float): Radial clearance c in mm.
        length (float): Bearing length l in mm.
        speed (float): Journal speed N in rev/s.
        load (float): Radial load W in N.
        viscosity (float): Lubricant absolute viscosity in mPa*s.
    """

    def __init__(
        self,
        radius,
        clearance,
        length,
        speed,
        load,
        viscosity=None,
        sae_grade=None,
        temperature=None,
        material="steel",
        name=None,
    ):
        """
        Initialize a JournalBearing object.

        The lubricant is specified either directly with ``viscosity`` or
        by ``sae_grade`` and ``temperature`` (resolved through the
        fig. 12-13 viscosity-temperature fit) — exactly one of the two.

        Args:
            radius (float): Journal radius r in mm.
            clearance (float): Radial clearance c in mm (must be smaller
                than the radius).
            length (float): Bearing length l in mm.
            speed (float): Journal speed N in rev/s.
            load (float): Radial load W in N.
            viscosity (float): Absolute viscosity in mPa*s.
            sae_grade (int): SAE oil grade (10-60), with ``temperature``.
            temperature (float): Mean film temperature in degrees Celsius,
                with ``sae_grade``.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the bearing.

        Raises:
            ValueError: For non-positive geometry/speed/load, a clearance
                not smaller than the radius, or an over- or
                under-specified lubricant.
        """
        super().__init__(name=name, material=material)
        if radius <= 0:
            raise ValueError("Radius must be strictly positive")
        if clearance <= 0:
            raise ValueError("Clearance must be strictly positive")
        if clearance >= radius:
            raise ValueError("Clearance must be smaller than the radius")
        if length <= 0:
            raise ValueError("Length must be strictly positive")
        if speed <= 0:
            raise ValueError("Speed must be strictly positive")
        if load <= 0:
            raise ValueError("Load must be strictly positive")
        if viscosity is not None and (sae_grade is not None or temperature is not None):
            raise ValueError(
                "Specify either viscosity or sae_grade+temperature, not both"
            )
        if viscosity is None:
            if sae_grade is None or temperature is None:
                raise ValueError(
                    "Lubricant unspecified: pass viscosity= or both "
                    "sae_grade= and temperature="
                )
            viscosity = sae_viscosity(sae_grade, temperature)
        if viscosity <= 0:
            raise ValueError("Viscosity must be strictly positive")
        self.radius = radius
        self.clearance = clearance
        self.length = length
        self.speed = speed
        self.load = load
        self.viscosity = viscosity

    @property
    def l_over_d(self):
        """Length-to-diameter ratio l/d."""
        return self.length / (2.0 * self.radius)

    @property
    def pressure(self):
        """Projected-area pressure P = W / (2 r l) in MPa."""
        return self.load / (2.0 * self.radius * self.length)

    @property
    def sommerfeld(self):
        """Sommerfeld (bearing characteristic) number S (eq. 12-7).

        S = (r/c)**2 * mu * N / P with mu in Pa*s, N in rev/s and P in Pa
        (converted internally from the stored mPa*s and MPa values).
        """
        mu_pa_s = self.viscosity * 1e-3
        pressure_pa = self.pressure * 1e6
        return (self.radius / self.clearance) ** 2 * mu_pa_s * self.speed / pressure_pa

    def petroff_friction(self):
        """Petroff friction coefficient (eq. 12-6, lightly loaded bearing).

        f = 2 * pi**2 * (mu * N / P) * (r / c).

        Returns:
            float: Coefficient of friction (dimensionless).
        """
        mu_pa_s = self.viscosity * 1e-3
        pressure_pa = self.pressure * 1e6
        return (
            2.0
            * math.pi**2
            * mu_pa_s
            * self.speed
            / pressure_pa
            * self.radius
            / self.clearance
        )

    def petroff_friction_torque(self):
        """Friction torque from the Petroff model, T = f * W * r in N*mm.

        Returns:
            float: Friction torque in N*mm.
        """
        return self.petroff_friction() * self.load * self.radius

    def is_thick_film(self):
        """Whether operation is in the stable thick-film regime (fig. 12-4).

        Returns:
            bool: True for hydrodynamic (thick-film) lubrication.
        """
        return is_thick_film(self.viscosity, self.speed, self.pressure)

    def performance(self):
        """Full Raimondi-Boyd performance analysis at the operating point.

        Interpolates the chart variables at the bearing's Sommerfeld
        number and l/d ratio, then converts them to physical quantities.

        Returns:
            dict: Physical results — ``h0`` minimum film thickness (mm),
            ``eccentricity_ratio`` (1 - h0/c), ``phi_deg`` position of
            minimum film (deg), ``friction_coefficient`` f,
            ``friction_torque`` (N*mm), ``power_loss`` (W), ``flow``
            total lubricant flow Q (mm^3/s), ``side_flow`` Qs (mm^3/s),
            ``pmax`` maximum film pressure (MPa) — plus the raw chart
            variables (``h0_over_c``, ``friction_variable``,
            ``flow_variable``, ``side_flow_ratio``, ``p_over_pmax``).
        """
        chart = raimondi_boyd(self.sommerfeld, self.l_over_d)
        h0 = chart["h0_over_c"] * self.clearance
        friction = chart["friction_variable"] * self.clearance / self.radius
        torque = friction * self.load * self.radius  # N*mm
        power_loss = 2.0 * math.pi * self.speed * torque / 1000.0  # W
        flow = (
            chart["flow_variable"]
            * self.radius
            * self.clearance
            * self.speed
            * self.length
        )  # mm^3/s
        result = dict(chart)
        result.update(
            {
                "h0": h0,
                "eccentricity_ratio": 1.0 - chart["h0_over_c"],
                "friction_coefficient": friction,
                "friction_torque": torque,
                "power_loss": power_loss,
                "flow": flow,
                "side_flow": chart["side_flow_ratio"] * flow,
                "pmax": self.pressure / chart["p_over_pmax"],
            }
        )
        return result

    def temperature_rise(self):
        """Lubricant temperature rise through the bearing (Shigley SI form).

        dT = 8.30 * P * (r/c)f / (Q/(r c N l) * (1 - 0.5 * Qs/Q)) with P
        in MPa, assuming all friction heat is carried away by the oil.

        Returns:
            float: Temperature rise in degrees Celsius.
        """
        chart = raimondi_boyd(self.sommerfeld, self.l_over_d)
        return (
            8.30
            * self.pressure
            * chart["friction_variable"]
            / (chart["flow_variable"] * (1.0 - 0.5 * chart["side_flow_ratio"]))
        )

    def trumpler_check(
        self, startup_load=None, max_temperature=None, design_factor=2.0
    ):
        """Evaluate the Trumpler design criteria (sec. 12-14).

        Criteria: minimum film h0 >= 0.005 mm + 0.00004 * d; maximum
        lubricant temperature <= 121 degC; starting pressure
        W_st / (2 r l) <= 2.07 MPa; and a design factor (default 2) on
        the running load still yielding a converged film.  The load
        criterion is reported as the minimum film check repeated at
        ``design_factor`` times the load.

        Args:
            startup_load (float): Static load at start-up in N (optional;
                check skipped as None when absent).
            max_temperature (float): Maximum lubricant temperature in
                degrees Celsius (optional; check skipped as None).
            design_factor (float): Load design factor n_d (default: 2.0).

        Returns:
            dict: ``min_film``, ``startup_pressure``, ``max_temperature``
            and ``design_factor_film`` mapped to True/False, or None for
            skipped checks.

        Raises:
            ValueError: For a non-positive startup load or design factor.
        """
        if design_factor <= 0:
            raise ValueError("Design factor must be strictly positive")
        diameter = 2.0 * self.radius
        h0_limit = TRUMPLER_MIN_FILM_BASE + TRUMPLER_MIN_FILM_SLOPE * diameter
        result = {
            "min_film": self.performance()["h0"] >= h0_limit,
            "startup_pressure": None,
            "max_temperature": None,
            "design_factor_film": None,
        }
        if startup_load is not None:
            if startup_load <= 0:
                raise ValueError("Startup load must be strictly positive")
            startup_pressure = startup_load / (2.0 * self.radius * self.length)
            result["startup_pressure"] = (
                startup_pressure <= TRUMPLER_MAX_STARTUP_PRESSURE
            )
        if max_temperature is not None:
            result["max_temperature"] = max_temperature <= TRUMPLER_MAX_TEMPERATURE
        if design_factor is not None:
            overloaded = JournalBearing(
                self.radius,
                self.clearance,
                self.length,
                self.speed,
                design_factor * self.load,
                viscosity=self.viscosity,
                material=self.material,
                name=self.name,
            )
            result["design_factor_film"] = overloaded.performance()["h0"] >= h0_limit
        return result

    def __repr__(self):
        return (
            f"JournalBearing(r={self.radius}, c={self.clearance}, "
            f"l={self.length}, N={self.speed} rev/s)"
        )
