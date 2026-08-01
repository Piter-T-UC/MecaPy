"""Hydrodynamic journal bearing analysis (Shigley ch. 12).

Units: geometry in mm, journal speed in rev/s, load in N, viscosity in
mPa*s, pressures in MPa, temperatures in degrees Celsius.

Dimensional inputs additionally accept a ``pint.Quantity`` (e.g.
``radius=1 * ureg.inch``, ``speed=1500 * ureg.rpm``), which is converted
to the documented unit at the boundary; plain floats are assumed to be in
that unit already and behave exactly as before.  Temperatures are
absolute (degC), not temperature differences.
"""

import math

from ..base import MechaElement
from ..utils.units import to_magnitude
from .lubrication_data import (
    TRUMPLER_MAX_STARTUP_PRESSURE,
    TRUMPLER_MAX_TEMPERATURE,
    TRUMPLER_MIN_FILM_BASE,
    TRUMPLER_MIN_FILM_SLOPE,
    WHIRL_ECCENTRICITY_THRESHOLD,
    WHIRL_FREQUENCY_RATIO,
    is_thick_film,
    raimondi_boyd,
    sommerfeld_for,
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
            radius (float): Journal radius r in mm (or a pint.Quantity of
                length).
            clearance (float): Radial clearance c in mm (must be smaller
                than the radius; or a pint.Quantity of length).
            length (float): Bearing length l in mm (or a pint.Quantity of
                length).
            speed (float): Journal speed N in rev/s (or a pint.Quantity
                of rotational speed, e.g. ``1500 * ureg.rpm``).
            load (float): Radial load W in N (or a pint.Quantity of force).
            viscosity (float): Absolute viscosity in mPa*s (or a
                pint.Quantity of dynamic viscosity).
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
            temperature = to_magnitude(temperature, "degC")
            viscosity = sae_viscosity(sae_grade, temperature)
        # Remember how the lubricant was specified: with a grade on file the
        # bearing can re-evaluate mu(T) itself (thermal solve); on the
        # explicit-viscosity path both stay None.
        self._sae_grade = sae_grade
        self._film_temperature = temperature
        # All six inputs go through the validating setters below, so mutating
        # any of them after construction re-checks it (and clearance < radius).
        self.radius = radius
        self.clearance = clearance
        self.length = length
        self.speed = speed
        self.load = load
        self.viscosity = viscosity

    # ---- Settable primary inputs (validate; keep clearance < radius) ----

    @property
    def radius(self):
        """float: Journal radius r in mm."""
        return self._radius

    @radius.setter
    def radius(self, value):
        value = to_magnitude(value, "mm")
        if value <= 0:
            raise ValueError("Radius must be strictly positive")
        if value <= getattr(self, "_clearance", 0.0):
            raise ValueError("Radius must be larger than the clearance")
        self._radius = value

    @property
    def clearance(self):
        """float: Radial clearance c in mm."""
        return self._clearance

    @clearance.setter
    def clearance(self, value):
        value = to_magnitude(value, "mm")
        if value <= 0:
            raise ValueError("Clearance must be strictly positive")
        if value >= self._radius:
            raise ValueError("Clearance must be smaller than the radius")
        self._clearance = value

    @property
    def length(self):
        """float: Bearing length l in mm."""
        return self._length

    @length.setter
    def length(self, value):
        value = to_magnitude(value, "mm")
        if value <= 0:
            raise ValueError("Length must be strictly positive")
        self._length = value

    @property
    def speed(self):
        """float: Journal speed N in rev/s."""
        return self._speed

    @speed.setter
    def speed(self, value):
        value = to_magnitude(value, "revolution / second")
        if value <= 0:
            raise ValueError("Speed must be strictly positive")
        self._speed = value

    @property
    def load(self):
        """float: Radial load W in N."""
        return self._load

    @load.setter
    def load(self, value):
        value = to_magnitude(value, "N")
        if value <= 0:
            raise ValueError("Load must be strictly positive")
        self._load = value

    @property
    def viscosity(self):
        """float: Absolute lubricant viscosity in mPa*s."""
        return self._viscosity

    @viscosity.setter
    def viscosity(self, value):
        value = to_magnitude(value, "mPa*s")
        if value <= 0:
            raise ValueError("Viscosity must be strictly positive")
        self._viscosity = value

    # ---- Lubricant provenance (read-only; set once at construction) ----

    @property
    def sae_grade(self):
        """int or None: SAE grade the viscosity was resolved from.

        ``None`` when the bearing was built with an explicit
        ``viscosity=``, in which case mu(T) cannot be re-evaluated.
        """
        return self._sae_grade

    @property
    def film_temperature(self):
        """float or None: Mean film temperature in degC the viscosity
        was evaluated at (``None`` on the explicit-viscosity path)."""
        return self._film_temperature

    @property
    def speed_rpm(self):
        """float: Journal speed in rev/min.

        Convenience view of :attr:`speed` (which the Sommerfeld/Petroff
        relations use in **rev/s**); ``speed_rpm = 60 * speed``.
        """
        return 60.0 * self._speed

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
        return self._performance_from_chart(
            raimondi_boyd(self.sommerfeld, self.l_over_d)
        )

    def _performance_from_chart(self, chart, load=None):
        """Physical performance from an already-interpolated chart dict.

        Split out so a caller holding a chart (``trumpler_check``) does
        not pay for a second interpolation.  ``load`` overrides the
        bearing's own load for the force-proportional entries.
        """
        load = self.load if load is None else load
        h0 = chart["h0_over_c"] * self.clearance
        friction = chart["friction_variable"] * self.clearance / self.radius
        torque = friction * load * self.radius  # N*mm
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
                "pmax": (load / (2.0 * self.radius * self.length))
                / chart["p_over_pmax"],
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
        return self._temperature_rise_from_chart(
            raimondi_boyd(self.sommerfeld, self.l_over_d)
        )

    def _temperature_rise_from_chart(self, chart, load=None):
        """Temperature rise from an already-interpolated chart dict."""
        load = self.load if load is None else load
        pressure = load / (2.0 * self.radius * self.length)
        return (
            8.30
            * pressure
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

        The overload branch uses the fact that with geometry, speed and
        viscosity fixed the Sommerfeld number scales as S ~ 1/W, so the
        chart is re-read at ``S / design_factor`` rather than a second
        bearing being built.

        Args:
            startup_load (float): Static load at start-up in N (optional;
                check skipped as None when absent; or a pint.Quantity of
                force).
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
        chart = raimondi_boyd(self.sommerfeld, self.l_over_d)
        overload_chart = raimondi_boyd(self.sommerfeld / design_factor, self.l_over_d)
        result = {
            "min_film": chart["h0_over_c"] * self.clearance >= h0_limit,
            "startup_pressure": None,
            "max_temperature": None,
            "design_factor_film": (
                overload_chart["h0_over_c"] * self.clearance >= h0_limit
            ),
        }
        if startup_load is not None:
            startup_load = to_magnitude(startup_load, "N")
            if startup_load <= 0:
                raise ValueError("Startup load must be strictly positive")
            startup_pressure = startup_load / (2.0 * self.radius * self.length)
            result["startup_pressure"] = (
                startup_pressure <= TRUMPLER_MAX_STARTUP_PRESSURE
            )
        if max_temperature is not None:
            result["max_temperature"] = max_temperature <= TRUMPLER_MAX_TEMPERATURE
        return result

    # ---- Thermal self-consistency (Shigley sec. 12-8) ----

    def solve_film_temperature(
        self,
        inlet_temperature,
        sae_grade=None,
        relaxation=0.5,
        tolerance=0.1,
        max_iterations=50,
        apply=True,
    ):
        """Solve the mean film temperature the bearing actually runs at.

        Viscosity and temperature are coupled: a trial temperature fixes
        mu(T), mu fixes the friction and so the temperature rise, and the
        rise fixes the temperature.  The design loop is the fixed point

        ::

            T_avg <- (1 - w) * T_avg + w * (T_in + dT(mu(SAE, T_avg)) / 2)

        which is a contraction because mu falls with T while dT rises
        with mu, so the composite map is decreasing; ``relaxation`` w
        damps the oscillation that the undamped update would show.

        Args:
            inlet_temperature (float): Lubricant inlet temperature in
                degrees Celsius (or a pint.Quantity).
            sae_grade (int): SAE oil grade; defaults to the grade the
                bearing was built with.
            relaxation (float): Damping factor w in (0, 1] (default: 0.5).
            tolerance (float): Convergence tolerance on the mean film
                temperature in degrees Celsius (default: 0.1).
            max_iterations (int): Iteration cap (default: 50).
            apply (bool): Write the converged viscosity and temperature
                back onto the bearing (default: True), so the object
                becomes its own self-consistent operating point.

        Returns:
            dict: ``temperature`` (mean film T_avg, degC),
            ``inlet_temperature``, ``outlet_temperature`` (T_in + dT),
            ``rise`` (dT, degC), ``viscosity`` (mPa*s), ``sommerfeld``,
            ``iterations`` and ``converged``.

        Raises:
            ValueError: If no SAE grade is available, or a parameter is
                out of range.
        """
        inlet_temperature = to_magnitude(inlet_temperature, "degC")
        grade = self._sae_grade if sae_grade is None else sae_grade
        if grade is None:
            raise ValueError(
                "Thermal solve needs an SAE grade; pass sae_grade= or build "
                "the bearing with sae_grade= and temperature="
            )
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("Relaxation must be in (0, 1]")
        if tolerance <= 0:
            raise ValueError("Tolerance must be strictly positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        # Chart terms depend on the viscosity only through S, so the loop
        # walks the trial temperature rather than mutating the bearing.
        pressure_pa = self.pressure * 1e6
        radius_ratio_squared = (self.radius / self.clearance) ** 2
        temperature = inlet_temperature
        converged = False
        iterations = 0
        rise = 0.0
        oil_viscosity = self.viscosity
        for iterations in range(1, max_iterations + 1):
            oil_viscosity = sae_viscosity(grade, temperature)
            sommerfeld = (
                radius_ratio_squared * oil_viscosity * 1e-3 * self.speed / pressure_pa
            )
            chart = raimondi_boyd(sommerfeld, self.l_over_d)
            rise = self._temperature_rise_from_chart(chart)
            updated = (1.0 - relaxation) * temperature + relaxation * (
                inlet_temperature + 0.5 * rise
            )
            converged = abs(updated - temperature) < tolerance
            temperature = updated
            if converged:
                break
        oil_viscosity = sae_viscosity(grade, temperature)
        if apply:
            self.viscosity = oil_viscosity  # validating setter
            self._sae_grade = grade
            self._film_temperature = temperature
        return {
            "temperature": temperature,
            "inlet_temperature": inlet_temperature,
            "outlet_temperature": inlet_temperature + rise,
            "rise": rise,
            "viscosity": oil_viscosity,
            "sommerfeld": (
                radius_ratio_squared * oil_viscosity * 1e-3 * self.speed / pressure_pa
            ),
            "iterations": iterations,
            "converged": converged,
        }

    # ---- Design inverses (size one unknown for a target film) ----

    @property
    def minimum_film_limit(self):
        """float: Trumpler minimum film 0.005 + 0.00004 * d in mm."""
        return TRUMPLER_MIN_FILM_BASE + TRUMPLER_MIN_FILM_SLOPE * 2.0 * self.radius

    def minimum_film_safety_factor(self):
        """Ratio of the actual minimum film to the Trumpler limit.

        Values greater than 1 mean the film clears the sec. 12-14
        criterion.

        Returns:
            float: h0 / (0.005 + 0.00004 * d), dimensionless.
        """
        return self.performance()["h0"] / self.minimum_film_limit

    def _sommerfeld_for_film(self, h0_target):
        """Sommerfeld number giving a target h0 at the current clearance."""
        h0_target = to_magnitude(h0_target, "mm")
        if h0_target <= 0:
            raise ValueError("Target film thickness must be strictly positive")
        if h0_target >= self.clearance:
            raise ValueError(
                "Target film thickness must be smaller than the radial clearance"
            )
        return sommerfeld_for(h0_target / self.clearance, self.l_over_d)

    def viscosity_for_minimum_film(self, h0_target):
        """Viscosity needed to hold a target minimum film thickness.

        Inverts the Sommerfeld definition at fixed geometry, speed and
        load: S = (r/c)^2 * mu * N / P is linear in mu, so once the chart
        gives the S that yields ``h0_target`` the viscosity follows in
        closed form (no iteration).

        Args:
            h0_target (float): Required minimum film thickness in mm (or
                a pint.Quantity of length); must be below the clearance.

        Returns:
            float: Required absolute viscosity in mPa*s.

        Raises:
            ValueError: If the target is non-positive, not below the
                clearance, or outside the chart range.
        """
        sommerfeld = self._sommerfeld_for_film(h0_target)
        pressure_pa = self.pressure * 1e6
        viscosity_pa_s = (
            sommerfeld
            * pressure_pa
            / ((self.radius / self.clearance) ** 2 * self.speed)
        )
        return viscosity_pa_s * 1e3

    def length_for_minimum_film(self, h0_target, tolerance=1e-9, max_iterations=100):
        """Bearing length needed to hold a target minimum film thickness.

        Lengthening the bearing drops the projected pressure and so
        raises S.  Unlike the viscosity inverse this is not a one-shot
        closed form, because changing l also changes l/d and therefore
        which chart is read; the closed-form step is iterated to a fixed
        point so the returned length really does deliver ``h0_target``.

        Args:
            h0_target (float): Required minimum film thickness in mm (or
                a pint.Quantity of length).
            tolerance (float): Relative convergence tolerance on the
                length (default: 1e-9).
            max_iterations (int): Iteration cap (default: 100).

        Returns:
            float: Required bearing length in mm.

        Raises:
            ValueError: As for :meth:`viscosity_for_minimum_film`, or if
                the iteration fails to converge.
        """
        h0_target = to_magnitude(h0_target, "mm")
        if h0_target <= 0:
            raise ValueError("Target film thickness must be strictly positive")
        if h0_target >= self.clearance:
            raise ValueError(
                "Target film thickness must be smaller than the radial clearance"
            )
        viscosity_pa_s = self.viscosity * 1e-3
        radius_ratio_squared = (self.radius / self.clearance) ** 2
        length = self.length
        for _ in range(max_iterations):
            sommerfeld = sommerfeld_for(
                h0_target / self.clearance, length / (2.0 * self.radius)
            )
            updated = (
                sommerfeld
                * 1e6
                * self.load
                / (
                    radius_ratio_squared
                    * viscosity_pa_s
                    * self.speed
                    * 2.0
                    * self.radius
                )
            )
            if abs(updated - length) <= tolerance * updated:
                return updated
            length = updated
        raise ValueError(
            "Length iteration did not converge; widen the target or raise "
            "max_iterations"
        )

    def film_for_clearance(self, clearance):
        """Minimum film thickness the bearing would have at a clearance.

        Pure function of the trial clearance — the bearing is not
        modified.  This is the trade-off behind clearance selection:
        tightening the clearance raises h0/c but shrinks c, so h0 peaks
        at an intermediate value.

        Args:
            clearance (float): Trial radial clearance in mm (or a
                pint.Quantity of length).

        Returns:
            float: Minimum film thickness h0 in mm.

        Raises:
            ValueError: If the clearance is not in (0, radius).
        """
        clearance = to_magnitude(clearance, "mm")
        if not 0.0 < clearance < self.radius:
            raise ValueError("Clearance must be strictly between 0 and the radius")
        sommerfeld = (
            (self.radius / clearance) ** 2
            * self.viscosity
            * 1e-3
            * self.speed
            / (self.pressure * 1e6)
        )
        return raimondi_boyd(sommerfeld, self.l_over_d)["h0_over_c"] * clearance

    def clearance_sweep(self, clearance_min, clearance_max, n=40):
        """Film thickness and losses across a range of clearances.

        Matplotlib-free design data, so it can be inspected and tested
        without a plotting backend (the same split as
        :meth:`~mecapy.gears.Transmission.stage_layout`).

        Args:
            clearance_min (float): Smallest clearance in mm.
            clearance_max (float): Largest clearance in mm.
            n (int): Number of samples (default: 40, minimum 2).

        Returns:
            dict: ``clearance``, ``h0``, ``sommerfeld``, ``power_loss``
            (W) and ``temperature_rise`` (degC), each a list of length n.

        Raises:
            ValueError: For a non-positive or inverted range, a clearance
                at or above the radius, or n < 2.
        """
        clearance_min = to_magnitude(clearance_min, "mm")
        clearance_max = to_magnitude(clearance_max, "mm")
        if clearance_min <= 0:
            raise ValueError("Clearance must be strictly positive")
        if clearance_max <= clearance_min:
            raise ValueError("clearance_max must exceed clearance_min")
        if clearance_max >= self.radius:
            raise ValueError("Clearance must be smaller than the radius")
        if n < 2:
            raise ValueError("n must be at least 2")
        result = {
            "clearance": [],
            "h0": [],
            "sommerfeld": [],
            "power_loss": [],
            "temperature_rise": [],
        }
        step = (clearance_max - clearance_min) / (n - 1)
        pressure_pa = self.pressure * 1e6
        for index in range(n):
            clearance = clearance_min + index * step
            sommerfeld = (
                (self.radius / clearance) ** 2
                * self.viscosity
                * 1e-3
                * self.speed
                / pressure_pa
            )
            chart = raimondi_boyd(sommerfeld, self.l_over_d)
            friction = chart["friction_variable"] * clearance / self.radius
            torque = friction * self.load * self.radius
            result["clearance"].append(clearance)
            result["h0"].append(chart["h0_over_c"] * clearance)
            result["sommerfeld"].append(sommerfeld)
            result["power_loss"].append(2.0 * math.pi * self.speed * torque / 1000.0)
            result["temperature_rise"].append(self._temperature_rise_from_chart(chart))
        return result

    def optimum_clearance(self, clearance_min, clearance_max, n=80):
        """Clearance that maximizes the minimum film thickness.

        h0(c) is unimodal (the Shigley clearance trade-off): a coarse
        scan brackets the peak and a golden-section refinement lands on
        it.  The bearing itself is not modified.

        Args:
            clearance_min (float): Smallest clearance to consider, mm.
            clearance_max (float): Largest clearance to consider, mm.
            n (int): Scan resolution (default: 80).

        Returns:
            float: Clearance in mm maximizing h0.

        Raises:
            ValueError: As for :meth:`clearance_sweep`.
        """
        sweep = self.clearance_sweep(clearance_min, clearance_max, n)
        best = max(range(n), key=lambda index: sweep["h0"][index])
        low = sweep["clearance"][max(best - 1, 0)]
        high = sweep["clearance"][min(best + 1, n - 1)]
        golden = 0.5 * (math.sqrt(5.0) - 1.0)
        for _ in range(200):
            if high - low < 1e-12:
                break
            probe_low = high - golden * (high - low)
            probe_high = low + golden * (high - low)
            if self.film_for_clearance(probe_low) < self.film_for_clearance(probe_high):
                low = probe_low
            else:
                high = probe_high
        return 0.5 * (low + high)

    def clearance_window_for_minimum_film(
        self, h0_target, clearance_min, clearance_max, n=200
    ):
        """Clearance range over which a target minimum film is met.

        Because h0(c) rises to a peak and falls again, an acceptable
        target is met over an interval, not at a point — that interval is
        the manufacturing tolerance the design can be built to.

        Args:
            h0_target (float): Required minimum film thickness in mm.
            clearance_min (float): Lower end of the search range, mm.
            clearance_max (float): Upper end of the search range, mm.
            n (int): Scan resolution (default: 200).

        Returns:
            tuple or None: ``(c_low, c_high)`` in mm, or None when the
            target is not reached anywhere in the range.

        Raises:
            ValueError: As for :meth:`clearance_sweep`.
        """
        h0_target = to_magnitude(h0_target, "mm")
        if h0_target <= 0:
            raise ValueError("Target film thickness must be strictly positive")
        sweep = self.clearance_sweep(clearance_min, clearance_max, n)
        meeting = [
            clearance
            for clearance, h0 in zip(sweep["clearance"], sweep["h0"])
            if h0 >= h0_target
        ]
        if not meeting:
            return None
        return min(meeting), max(meeting)

    # ---- Stability (half-frequency whirl) ----

    @property
    def eccentricity_ratio(self):
        """float: Eccentricity ratio eps = 1 - h0/c, from 0 (concentric)
        to 1 (journal touching the bush)."""
        return 1.0 - raimondi_boyd(self.sommerfeld, self.l_over_d)["h0_over_c"]

    @property
    def is_whirl_prone(self):
        """bool: True when the journal runs too lightly loaded to be
        stable against half-frequency whirl.

        A lightly loaded journal sits near the center of its clearance
        circle, and the oil film's cross-coupled stiffness can drive the
        journal into an orbit at roughly half shaft speed.  The practical
        rule used here is that whirl is a risk below an eccentricity
        ratio of :data:`WHIRL_ECCENTRICITY_THRESHOLD` (0.6); this is a
        rotordynamics rule of thumb, **not** a Shigley result, and a real
        stability assessment needs the film's stiffness and damping
        coefficients together with the rotor mass.
        """
        return self.eccentricity_ratio < WHIRL_ECCENTRICITY_THRESHOLD

    @property
    def whirl_margin(self):
        """float: Eccentricity ratio over the whirl threshold (0.6).

        Greater than 1 means the journal is loaded firmly enough to be
        outside the usual whirl-prone region.
        """
        return self.eccentricity_ratio / WHIRL_ECCENTRICITY_THRESHOLD

    @property
    def whirl_frequency(self):
        """float: Approximate whirl frequency in rev/s.

        Half-frequency whirl runs near
        :data:`WHIRL_FREQUENCY_RATIO` (0.47) times journal speed.
        """
        return WHIRL_FREQUENCY_RATIO * self.speed

    # ---- Film geometry, reports and plots ----

    def film_profile(self, n=181):
        """Film thickness around the circumference.

        ::

            h(theta) = c * (1 + eps * cos(theta))

        with theta measured from the point of maximum film, so the
        minimum film h0 = c * (1 - eps) falls at theta = 180 degrees.
        Matplotlib-free, so the geometry behind :meth:`plot_film` can be
        tested directly.

        Args:
            n (int): Number of samples around the circumference
                (default: 181, i.e. one per 2 degrees).

        Returns:
            dict: ``angle_deg`` and ``film_thickness`` (mm) lists of
            length n, plus the scalars ``eccentricity_ratio``,
            ``clearance``, ``h0`` (mm) and ``phi_deg`` (attitude angle).

        Raises:
            ValueError: If n is less than 2.
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        chart = raimondi_boyd(self.sommerfeld, self.l_over_d)
        eccentricity = 1.0 - chart["h0_over_c"]
        angles = [360.0 * index / (n - 1) for index in range(n)]
        thickness = [
            self.clearance * (1.0 + eccentricity * math.cos(math.radians(angle)))
            for angle in angles
        ]
        return {
            "angle_deg": angles,
            "film_thickness": thickness,
            "eccentricity_ratio": eccentricity,
            "clearance": self.clearance,
            "h0": chart["h0_over_c"] * self.clearance,
            "phi_deg": chart["phi_deg"],
        }

    def describe(self):
        """
        Human-readable summary of geometry, operating point and checks.

        The string is returned, not printed; use
        ``print(journal.describe())``.

        Returns:
            str: Multi-line description, one quantity per line as
            ``label (symbol) = value unit``.
        """
        performance = self.performance()
        header = f"{self.__class__.__name__} geometry"
        if self.name:
            header += f" '{self.name}'"
        lines = [
            header,
            "=" * 40,
            f"journal radius (r) = {self.radius:.3f} mm",
            f"radial clearance (c) = {self.clearance:.4f} mm",
            f"bearing length (l) = {self.length:.3f} mm",
            f"length ratio (l/d) = {self.l_over_d:.3f}",
            f"radius ratio (r/c) = {self.radius / self.clearance:.1f}",
            f"speed (N) = {self.speed:.3f} rev/s ({self.speed_rpm:.1f} rpm)",
            f"radial load (W) = {self.load:.1f} N",
            f"unit pressure (P) = {self.pressure:.4f} MPa",
            f"viscosity (mu) = {self.viscosity:.2f} mPa*s",
        ]
        if self.sae_grade is not None:
            lines.append(
                f"lubricant = SAE {self.sae_grade} at "
                f"{self.film_temperature:.1f} degC"
            )
        lines += [
            f"Sommerfeld number (S) = {self.sommerfeld:.4f}",
            f"minimum film (h0) = {performance['h0']:.4f} mm",
            f"eccentricity ratio (eps) = {performance['eccentricity_ratio']:.3f}",
            f"attitude angle (phi) = {performance['phi_deg']:.1f} deg",
            f"friction coefficient (f) = {performance['friction_coefficient']:.5f}",
            f"power loss (H) = {performance['power_loss']:.1f} W",
            f"total flow (Q) = {performance['flow']:.1f} mm^3/s",
            f"side flow ratio (Qs/Q) = {performance['side_flow_ratio']:.3f}",
            f"peak film pressure (pmax) = {performance['pmax']:.3f} MPa",
            f"temperature rise (dT) = {self.temperature_rise():.1f} degC",
            f"thick film (fig. 12-4) = {'yes' if self.is_thick_film() else 'no'}",
            f"minimum film safety = {self.minimum_film_safety_factor():.2f}",
            f"whirl margin = {self.whirl_margin:.2f}"
            f" ({'whirl-prone' if self.is_whirl_prone else 'stable'})",
            f"material = {self.material}",
        ]
        return "\n".join(lines)

    def plot_film(self, show=True, ax=None, exaggeration=None):
        """Draw the journal in its bore with the film to scale.

        Clearances are three orders of magnitude smaller than the radius,
        so the eccentricity is drawn at an exaggerated scale (chosen
        automatically unless ``exaggeration`` is given) while the film
        thickness annotation reports the true value.

        Args:
            show (bool): Call ``plt.show()`` before returning
                (default: True).
            ax (matplotlib.axes.Axes): Axes to draw on; a new figure is
                created when omitted.
            exaggeration (float): Clearance magnification factor.

        Returns:
            matplotlib.figure.Figure: The figure drawn on.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_film; "
                "install it with 'pip install matplotlib'"
            )
        profile = self.film_profile()
        if exaggeration is None:
            exaggeration = 0.25 * self.radius / self.clearance
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure
        drawn_clearance = self.clearance * exaggeration
        bore_radius = self.radius + drawn_clearance
        offset = profile["eccentricity_ratio"] * drawn_clearance
        attitude = math.radians(profile["phi_deg"])
        center = (-offset * math.sin(attitude), -offset * math.cos(attitude))
        ax.add_patch(
            Circle((0.0, 0.0), bore_radius, facecolor="#dbeafe", edgecolor="#6b7280")
        )
        ax.add_patch(
            Circle(center, self.radius, facecolor="white", edgecolor="#1d4ed8")
        )
        ax.plot([center[0]], [center[1]], marker="+", color="#1d4ed8")
        ax.plot([0.0], [0.0], marker="+", color="#6b7280")
        ax.annotate(
            f"h0 = {profile['h0'] * 1000:.1f} um at phi = {profile['phi_deg']:.0f} deg",
            xy=(0.5, 0.02),
            xycoords="axes fraction",
            ha="center",
            color="#1d4ed8",
        )
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(
            f"Journal in bore (clearance x{exaggeration:.0f}), "
            f"eps = {profile['eccentricity_ratio']:.2f}"
        )
        ax.set_xlim(-1.15 * bore_radius, 1.15 * bore_radius)
        ax.set_ylim(-1.15 * bore_radius, 1.15 * bore_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        if show:
            plt.show()
        return fig

    def plot_clearance_design(
        self, clearance_min=None, clearance_max=None, n=40, show=True, ax=None
    ):
        """Plot minimum film and power loss against radial clearance.

        The classic clearance trade-off: h0 peaks at an intermediate
        clearance while the power loss falls monotonically, so the
        designer picks a window rather than a point.

        Args:
            clearance_min (float): Lower end of the sweep in mm
                (default: a quarter of the current clearance).
            clearance_max (float): Upper end in mm (default: four times
                the current clearance).
            n (int): Number of samples (default: 40).
            show (bool): Call ``plt.show()`` before returning.
            ax (matplotlib.axes.Axes): Axes to draw on.

        Returns:
            matplotlib.figure.Figure: The figure drawn on.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_clearance_design; "
                "install it with 'pip install matplotlib'"
            )
        if clearance_min is None:
            clearance_min = 0.25 * self.clearance
        if clearance_max is None:
            clearance_max = 4.0 * self.clearance
        sweep = self.clearance_sweep(clearance_min, clearance_max, n)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure
        micron = [value * 1000.0 for value in sweep["h0"]]
        ax.plot(sweep["clearance"], micron, color="#1d4ed8", label="minimum film h0")
        ax.axhline(
            self.minimum_film_limit * 1000.0,
            color="#6b7280",
            linestyle="--",
            label="Trumpler limit",
        )
        ax.axvline(self.clearance, color="#6b7280", linewidth=0.8)
        ax.set_xlabel("radial clearance c [mm]")
        ax.set_ylabel("minimum film h0 [um]")
        power_axis = ax.twinx()
        power_axis.plot(
            sweep["clearance"],
            sweep["power_loss"],
            color="#dbeafe",
            label="power loss",
        )
        power_axis.set_ylabel("power loss [W]")
        ax.set_title("Clearance trade-off")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right")
        if show:
            plt.show()
        return fig

    def __repr__(self):
        return (
            f"JournalBearing(r={self.radius}, c={self.clearance}, "
            f"l={self.length}, N={self.speed} rev/s)"
        )
