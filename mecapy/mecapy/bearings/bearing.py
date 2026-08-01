"""Rolling-contact bearing design and analysis module (Shigley ch. 11).

Units: dimensions in mm, forces and load ratings in N, life in revolutions
unless noted otherwise.  Catalog ratings quoted in kN must be multiplied
by 1000 before being passed in.

Dimensional inputs additionally accept a ``pint.Quantity`` (e.g.
``bore_diameter=1 * ureg.inch``, ``C10=35 * ureg.kN``), which is converted
to the documented unit at the boundary; plain floats are assumed to be in
that unit already and behave exactly as before.
"""

import math

from ..base import MechaElement
from ..utils.units import to_magnitude
from .bearing_data import (
    DEFAULT_RATING_LIFE,
    ROTATION_FACTORS,
    WEIBULL_B,
    WEIBULL_THETA,
    WEIBULL_X0,
    XY_TABLE_TYPES,
    get_life_exponent,
    get_xy_factors,
    weibull_life_multiplier,
    weibull_reliability,
)
from .iso281_data import (
    DEFAULT_OIL_DENSITY,
    FATIGUE_LOAD_LIMIT_RATIO,
    TAPERED_K_DEFAULT,
    a_iso,
    get_a_iso_family,
    get_contamination_factor,
    get_limiting_dn,
    get_reliability_factor,
    get_static_factors,
    kinematic_from_dynamic,
    viscosity_ratio,
)
from .lubrication_data import viscosity as sae_viscosity


class Bearing(MechaElement):
    """
    Rolling-contact bearing design and analysis.

    Implements the Shigley ch. 11 rating-life model: load-life relation
    (eq. 11-1), Weibull reliability adjustment (eq. 11-5), catalog sizing
    (eq. 11-9/11-10), equivalent radial load under combined loading
    (table 11-1, eq. 11-12) and duty-cycle (cubic mean) loading
    (sec. 11-10).

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`.

    Attributes:
        bore_diameter (float): Bearing bore diameter in mm.
        outer_diameter (float): Bearing outer diameter in mm.
        width (float): Bearing width in mm.
        bearing_type (str): Type of bearing.
        C10 (float): Basic dynamic load rating in N (or None).
        C0 (float): Static load rating in N (or None).
        rating_life (float): Life associated with C10 in revolutions.
    """

    def __init__(
        self,
        bore_diameter,
        outer_diameter,
        width,
        bearing_type="ball",
        material="steel",
        name=None,
        C10=None,
        C0=None,
        rating_life=DEFAULT_RATING_LIFE,
    ):
        """
        Initialize a Bearing object.

        Args:
            bore_diameter (float): Bearing bore diameter in mm (or a
                pint.Quantity of length).
            outer_diameter (float): Bearing outer diameter in mm (or a
                pint.Quantity of length).
            width (float): Bearing width in mm (or a pint.Quantity of
                length).
            bearing_type (str): Type of bearing (default: "ball").
                One of "ball", "angular_contact", "roller", "cylindrical",
                "tapered"; sets the load-life exponent (3 or 10/3).
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the bearing.
            C10 (float): Basic dynamic load rating in N (or a
                pint.Quantity of force -- ``26 * ureg.kN`` saves the
                conversion).  Plain catalog values in kN must be
                multiplied by 1000.  Optional; life methods require it.
            C0 (float): Static load rating in N (same unit note as C10).
                Optional; combined-load methods require it.
            rating_life (float): Life associated with C10 in revolutions
                (default: 1e6, the usual catalog basis).

        Raises:
            ValueError: For non-positive dimensions or ratings, an outer
                diameter not larger than the bore, or an unknown bearing
                type.
        """
        super().__init__(name=name, material=material)
        # Every input is routed through a validating property setter below,
        # so mutating a dimension or rating after construction re-checks it
        # (and, for the diameters, the outer > bore invariant).
        self.bore_diameter = bore_diameter
        self.outer_diameter = outer_diameter
        self.width = width
        self.bearing_type = bearing_type  # validating setter rejects bad types
        self.C10 = C10
        self.C0 = C0
        self.rating_life = rating_life

    # ---- Settable primary inputs (validate; never cache a derived value) ----

    @property
    def bore_diameter(self):
        """float: Bearing bore diameter in mm."""
        return self._bore_diameter

    @bore_diameter.setter
    def bore_diameter(self, value):
        value = to_magnitude(value, "mm")
        if value <= 0:
            raise ValueError("Bore diameter must be strictly positive")
        if value >= getattr(self, "_outer_diameter", math.inf):
            raise ValueError("Bore diameter must be smaller than outer diameter")
        self._bore_diameter = value

    @property
    def outer_diameter(self):
        """float: Bearing outer diameter in mm."""
        return self._outer_diameter

    @outer_diameter.setter
    def outer_diameter(self, value):
        value = to_magnitude(value, "mm")
        if value <= self._bore_diameter:
            raise ValueError("Outer diameter must be larger than bore diameter")
        self._outer_diameter = value

    @property
    def width(self):
        """float: Bearing width in mm."""
        return self._width

    @width.setter
    def width(self, value):
        value = to_magnitude(value, "mm")
        if value <= 0:
            raise ValueError("Width must be strictly positive")
        self._width = value

    @property
    def C10(self):
        """float or None: Basic dynamic load rating in N."""
        return self._C10

    @C10.setter
    def C10(self, value):
        if value is not None:
            value = to_magnitude(value, "N")
            if value <= 0:
                raise ValueError("Dynamic load rating C10 must be strictly positive")
        self._C10 = value

    @property
    def C0(self):
        """float or None: Static load rating in N."""
        return self._C0

    @C0.setter
    def C0(self, value):
        if value is not None:
            value = to_magnitude(value, "N")
            if value <= 0:
                raise ValueError("Static load rating C0 must be strictly positive")
        self._C0 = value

    @property
    def rating_life(self):
        """float: Life associated with C10 in revolutions."""
        return self._rating_life

    @rating_life.setter
    def rating_life(self, value):
        value = to_magnitude(value, "revolution")
        if value <= 0:
            raise ValueError("Rating life must be strictly positive")
        self._rating_life = value

    @property
    def bearing_type(self):
        """str: Bearing type (sets the load-life exponent)."""
        return self._bearing_type

    @bearing_type.setter
    def bearing_type(self, value):
        get_life_exponent(value)  # raises ValueError on an unknown type
        self._bearing_type = value

    @property
    def life_exponent(self):
        """float: Load-life exponent a (3 for ball, 10/3 for roller).

        Derived from :attr:`bearing_type`, so changing the type updates it
        (and every life result) on the next access — never cached.
        """
        return get_life_exponent(self.bearing_type)

    def _require_c10(self):
        if self.C10 is None:
            raise ValueError("Bearing has no C10 rating; pass C10= to the constructor")

    def life(self, load, application_factor=1.0):
        """Rating (L10) life at a steady radial load (eq. 11-1/11-3).

        L = rating_life * (C10 / (af * load))**a, with a = 3 for ball
        and 10/3 for roller bearings.

        Args:
            load (float): Steady equivalent radial load in N (or a
                pint.Quantity of force).
            application_factor (float): Load-application factor af
                (table 11-5, default: 1.0).

        Returns:
            float: Life in revolutions with 90% reliability.

        Raises:
            ValueError: If the load or factor is non-positive, or the
                bearing has no C10 rating.
        """
        self._require_c10()
        load = to_magnitude(load, "N")
        if load <= 0:
            raise ValueError("Load must be strictly positive")
        if application_factor <= 0:
            raise ValueError("Application factor must be strictly positive")
        return (
            self.rating_life
            * (self.C10 / (application_factor * load)) ** self.life_exponent
        )

    def life_hours(self, load, speed, application_factor=1.0):
        """Rating life in hours at a steady load and speed.

        Args:
            load (float): Steady equivalent radial load in N (or a
                pint.Quantity of force).
            speed (float): Shaft speed in rpm (or a pint.Quantity of
                rotational speed).
            application_factor (float): Load-application factor af
                (default: 1.0).

        Returns:
            float: Life in hours with 90% reliability.

        Raises:
            ValueError: If the speed is non-positive (plus the checks
                of :meth:`life`).
        """
        speed = to_magnitude(speed, "revolutions_per_minute")
        if speed <= 0:
            raise ValueError("Speed must be strictly positive")
        return self.life(load, application_factor) / (60.0 * speed)

    def adjusted_life(self, load, reliability=0.90, application_factor=1.0):
        """Life at a reliability other than 90% (eq. 11-6, Weibull model).

        Args:
            load (float): Steady equivalent radial load in N.
            reliability (float): Desired probability of survival,
                0 < R < 1 (default: 0.90).
            application_factor (float): Load-application factor af
                (default: 1.0).

        Returns:
            float: Life in revolutions at the given reliability.
        """
        return self.life(load, application_factor) * weibull_life_multiplier(
            reliability
        )

    def reliability(self, load, life_revolutions, application_factor=1.0):
        """Probability of surviving a given life at a load (eq. 11-5).

        Args:
            load (float): Steady equivalent radial load in N.
            life_revolutions (float): Required life in revolutions.
            application_factor (float): Load-application factor af
                (default: 1.0).

        Returns:
            float: Probability of survival, 0 < R <= 1.

        Raises:
            ValueError: If the required life is negative.
        """
        life_revolutions = to_magnitude(life_revolutions, "revolution")
        if life_revolutions < 0:
            raise ValueError("Life must be non-negative")
        x = life_revolutions / self.life(load, application_factor)
        return weibull_reliability(x)

    def required_C10(
        self,
        load,
        desired_life,
        reliability=0.90,
        application_factor=1.0,
        approximate=False,
    ):
        """Catalog rating needed for a life and reliability (eq. 11-9/11-10).

        C10 = af * F * (xD / (x0 + (theta - x0) * z))**(1/a) with
        xD = desired_life / rating_life and z = (ln(1/R))**(1/b) exactly,
        or z = (1 - R)**(1/b) when ``approximate`` is True (the textbook
        eq. 11-10 linearization, accurate for R >= 0.90).

        Args:
            load (float): Design radial load in N (or a pint.Quantity of
                force).
            desired_life (float): Required life in revolutions (or a
                pint.Quantity of angle/revolutions).
            reliability (float): Required probability of survival,
                0 < R < 1 (default: 0.90).
            application_factor (float): Load-application factor af
                (default: 1.0).
            approximate (bool): Use the eq. 11-10 approximation for z
                (default: False, exact Weibull inversion).

        Returns:
            float: Required basic dynamic load rating C10 in N.

        Raises:
            ValueError: For non-positive load/life/factor or reliability
                outside (0, 1).
        """
        load = to_magnitude(load, "N")
        desired_life = to_magnitude(desired_life, "revolution")
        if load <= 0:
            raise ValueError("Load must be strictly positive")
        if desired_life <= 0:
            raise ValueError("Desired life must be strictly positive")
        if application_factor <= 0:
            raise ValueError("Application factor must be strictly positive")
        if not 0.0 < reliability < 1.0:
            raise ValueError("Reliability must be strictly between 0 and 1")
        x_d = desired_life / self.rating_life
        if approximate:
            z = (1.0 - reliability) ** (1.0 / WEIBULL_B)
        else:
            z = math.log(1.0 / reliability) ** (1.0 / WEIBULL_B)
        x_r = WEIBULL_X0 + (WEIBULL_THETA - WEIBULL_X0) * z
        return application_factor * load * (x_d / x_r) ** (1.0 / self.life_exponent)

    def equivalent_load(self, radial_load, axial_load=0.0, rotating="inner"):
        """Equivalent radial load under combined loading (table 11-1).

        P = X * V * Fr + Y * Fa (eq. 11-12) with X, Y interpolated from
        table 11-1 on Fa/C0, but never less than V * Fr.  The table
        applies to single-row ball bearings; roller bearings with axial
        load need manufacturer data and are rejected.

        Args:
            radial_load (float): Radial load Fr in N (or a pint.Quantity
                of force).
            axial_load (float): Axial (thrust) load Fa in N (default: 0;
                or a pint.Quantity of force).
            rotating (str): Which ring rotates, "inner" or "outer"
                (sets V = 1.0 or 1.2, default: "inner").

        Returns:
            float: Equivalent steady radial load P in N.

        Raises:
            ValueError: For negative loads, an unknown rotating ring, a
                roller-family bearing with axial load, or a missing C0
                rating when axial load is present.
        """
        radial_load = to_magnitude(radial_load, "N")
        axial_load = to_magnitude(axial_load, "N")
        if radial_load < 0 or axial_load < 0:
            raise ValueError("Loads must be non-negative")
        if rotating not in ROTATION_FACTORS:
            raise ValueError(
                f"Unknown rotating ring '{rotating}'. "
                f"Available: {sorted(ROTATION_FACTORS)}"
            )
        v = ROTATION_FACTORS[rotating]
        if axial_load == 0:
            return v * radial_load
        if self.bearing_type not in XY_TABLE_TYPES:
            raise ValueError(
                "Table 11-1 X/Y factors apply to ball bearings only; "
                "roller bearings under axial load need manufacturer data"
            )
        if self.C0 is None:
            raise ValueError(
                "Combined loading requires the static rating C0; "
                "pass C0= to the constructor"
            )
        e, x2, y2 = get_xy_factors(axial_load / self.C0)
        if radial_load > 0 and axial_load / (v * radial_load) <= e:
            return v * radial_load
        return max(x2 * v * radial_load + y2 * axial_load, v * radial_load)

    def equivalent_steady_load(self, duty_cycle):
        """Cubic-mean load for variable loading (sec. 11-10, eq. 11-15).

        Feq = (sum(f_i * n_i * (af_i * F_i)**a) / sum(f_i * n_i))**(1/a)
        where f_i is the time fraction, n_i the speed and af_i an
        optional per-segment application factor.

        Args:
            duty_cycle (list): Segments as tuples of
                ``(load_N, time_fraction)``,
                ``(load_N, time_fraction, speed_rpm)`` or
                ``(load_N, time_fraction, speed_rpm, application_factor)``.
                Omitted speeds weight all segments equally.  Loads and
                speeds may individually be pint Quantities.

        Returns:
            float: Equivalent steady load in N producing the same damage.

        Raises:
            ValueError: For an empty duty cycle, malformed segments or
                non-positive loads, fractions, speeds or factors.
        """
        if not duty_cycle:
            raise ValueError("Duty cycle must contain at least one segment")
        a = self.life_exponent
        numerator = 0.0
        denominator = 0.0
        for segment in duty_cycle:
            if len(segment) == 2:
                load, fraction = segment
                speed, af = 1.0, 1.0
            elif len(segment) == 3:
                load, fraction, speed = segment
                af = 1.0
            elif len(segment) == 4:
                load, fraction, speed, af = segment
            else:
                raise ValueError(
                    "Duty-cycle segments must be (load, fraction[, speed[, af]])"
                )
            load = to_magnitude(load, "N")
            speed = to_magnitude(speed, "revolutions_per_minute")
            if load <= 0 or fraction <= 0 or speed <= 0 or af <= 0:
                raise ValueError(
                    "Loads, fractions, speeds and factors must be strictly positive"
                )
            weight = fraction * speed
            numerator += weight * (af * load) ** a
            denominator += weight
        return (numerator / denominator) ** (1.0 / a)

    # ---- ISO 281 modified rating life (a1 * a_ISO * L10) ----

    @property
    def mean_diameter(self):
        """float: Mean diameter dm = (d + D) / 2 in mm.

        The reference diameter of both the ISO 281 viscosity relations
        and the limiting-speed n*dm product.
        """
        return 0.5 * (self.bore_diameter + self.outer_diameter)

    def fatigue_load_limit(self, Cu=None):
        """Fatigue load limit Cu in N.

        An explicit catalog ``Cu`` always wins.  Without one, the common
        manufacturer rule of thumb Cu ~ C0 / 8.2 is used and is only an
        estimate — a real selection should take Cu from the catalog page.

        Args:
            Cu (float): Catalog fatigue load limit in N (or a
                pint.Quantity of force).  Optional.

        Returns:
            float: Fatigue load limit Cu in N.

        Raises:
            ValueError: If ``Cu`` is non-positive, or it is omitted and
                the bearing has no C0 rating to estimate from.
        """
        if Cu is not None:
            Cu = to_magnitude(Cu, "N")
            if Cu <= 0:
                raise ValueError("Fatigue load limit Cu must be strictly positive")
            return Cu
        if self.C0 is None:
            raise ValueError(
                "Estimating Cu needs the static rating C0; pass Cu= or C0="
            )
        return self.C0 / FATIGUE_LOAD_LIMIT_RATIO

    def viscosity_ratio(
        self,
        speed,
        kinematic_viscosity=None,
        sae_grade=None,
        temperature=None,
        density=DEFAULT_OIL_DENSITY,
    ):
        """Viscosity ratio kappa = nu / nu1 at an operating point.

        The lubricant is given either directly as ``kinematic_viscosity``
        or as ``sae_grade`` plus ``temperature``, which is resolved
        through the same Seireg-Dandage fit the journal bearings use and
        converted from mPa*s to mm^2/s with ``density``.  An explicit
        viscosity always wins.

        Args:
            speed (float): Operating speed in rpm (or a pint.Quantity).
            kinematic_viscosity (float): Operating viscosity nu in mm^2/s
                (= cSt).
            sae_grade (int): SAE oil grade (10-60), with ``temperature``.
            temperature (float): Oil temperature in degrees Celsius, with
                ``sae_grade``.
            density (float): Lubricant density in kg/m^3 (default: 870).

        Returns:
            float: Viscosity ratio kappa (unclamped).

        Raises:
            ValueError: If the lubricant is over- or under-specified, or
                any input is non-positive.
        """
        speed = to_magnitude(speed, "revolutions_per_minute")
        if kinematic_viscosity is not None and (
            sae_grade is not None or temperature is not None
        ):
            raise ValueError(
                "Specify either kinematic_viscosity or sae_grade+temperature, "
                "not both"
            )
        if kinematic_viscosity is None:
            if sae_grade is None or temperature is None:
                raise ValueError(
                    "Lubricant unspecified: pass kinematic_viscosity= or both "
                    "sae_grade= and temperature="
                )
            kinematic_viscosity = kinematic_from_dynamic(
                sae_viscosity(sae_grade, temperature), density=density
            )
        return viscosity_ratio(kinematic_viscosity, self.mean_diameter, speed)

    def life_modification_factor(
        self,
        load,
        speed,
        contamination="normal_cleanliness",
        contamination_position="mid",
        Cu=None,
        **lubricant,
    ):
        """ISO 281 life modification factor a_ISO at an operating point.

        Combines the cleanliness of the installation (e_C), the fatigue
        load limit (Cu) and the film quality (kappa) into the single
        multiplier that separates the basic rating life from the modified
        one.

        Args:
            load (float): Equivalent dynamic load P in N (or a
                pint.Quantity of force).
            speed (float): Operating speed in rpm (or a pint.Quantity).
            contamination (str): Cleanliness level, a key of
                :data:`~mecapy.bearings.iso281_data.CONTAMINATION_LEVELS`
                (default: "normal_cleanliness").
            contamination_position (str): "min", "mid" or "max" within
                the tabulated e_C band (default: "mid").
            Cu (float): Fatigue load limit in N; estimated from C0 when
                omitted (see :meth:`fatigue_load_limit`).
            lubricant: Extra keyword arguments passed to
                :meth:`viscosity_ratio` — ``kinematic_viscosity=`` or
                ``sae_grade=`` plus ``temperature=`` (and optionally
                ``density=``).

        Returns:
            float: Life modification factor a_ISO, between 0.1 and 50.

        Raises:
            ValueError: If the load is non-positive or any lookup fails.
        """
        load = to_magnitude(load, "N")
        if load <= 0:
            raise ValueError("Load must be strictly positive")
        kappa = self.viscosity_ratio(speed, **lubricant)
        e_c = get_contamination_factor(
            contamination, self.mean_diameter, position=contamination_position
        )
        load_ratio = e_c * self.fatigue_load_limit(Cu) / load
        return a_iso(load_ratio, kappa, family=get_a_iso_family(self.bearing_type))

    def iso_life(
        self,
        load,
        speed,
        reliability=0.90,
        contamination="normal_cleanliness",
        contamination_position="mid",
        Cu=None,
        application_factor=1.0,
        **lubricant,
    ):
        """Modified rating life L_nm in revolutions (ISO 281:2007).

        ::

            L_nm = a1 * a_ISO * L10                    (ISO 281 clause 6)

        This is the ISO path and is deliberately distinct from
        :meth:`adjusted_life`, which applies Shigley's 3-parameter
        Weibull model to the same L10.  The two use different reliability
        conventions and will not agree numerically.

        Args:
            load (float): Equivalent dynamic load P in N.
            speed (float): Operating speed in rpm.
            reliability (float): Probability of survival, 0.90 to 0.999
                (default: 0.90, where a1 = 1).
            contamination (str): Cleanliness level (default:
                "normal_cleanliness").
            contamination_position (str): "min", "mid" or "max".
            Cu (float): Fatigue load limit in N (estimated when omitted).
            application_factor (float): Load-application factor af.
            lubricant: Extra keyword arguments, see
                :meth:`viscosity_ratio`.

        Returns:
            float: Modified rating life in revolutions.

        Raises:
            ValueError: Per the underlying lookups and :meth:`life`.
        """
        a_1 = get_reliability_factor(reliability)
        a_iso_factor = self.life_modification_factor(
            load,
            speed,
            contamination=contamination,
            contamination_position=contamination_position,
            Cu=Cu,
            **lubricant,
        )
        return a_1 * a_iso_factor * self.life(load, application_factor)

    def iso_life_hours(self, load, speed, **kwargs):
        """Modified rating life in hours (see :meth:`iso_life`).

        Args:
            load (float): Equivalent dynamic load P in N.
            speed (float): Operating speed in rpm.
            kwargs: Extra keyword arguments passed straight to
                :meth:`iso_life`.

        Returns:
            float: Modified rating life in hours.
        """
        speed_rpm = to_magnitude(speed, "revolutions_per_minute")
        if speed_rpm <= 0:
            raise ValueError("Speed must be strictly positive")
        return self.iso_life(load, speed, **kwargs) / (60.0 * speed_rpm)

    # ---- Static rating (ISO 76) and speed limits ----

    def static_equivalent_load(
        self, radial_load, axial_load=0.0, contact_angle_deg=None
    ):
        """Static equivalent load P0 in N (ISO 76).

        ::

            P0 = max(X0 * Fr + Y0 * Fa, Fr)              (ISO 76 clause 7)

        The floor matters: for a lightly thrust-loaded deep-groove ball
        bearing the plain radial load governs.

        Args:
            radial_load (float): Radial load Fr in N (or a pint.Quantity).
            axial_load (float): Axial load Fa in N (default: 0).
            contact_angle_deg (float): Contact angle in degrees, required
                for the angular-contact and tapered families.

        Returns:
            float: Static equivalent load P0 in N.

        Raises:
            ValueError: For negative loads or a missing contact angle.
        """
        radial_load = to_magnitude(radial_load, "N")
        axial_load = to_magnitude(axial_load, "N")
        if radial_load < 0 or axial_load < 0:
            raise ValueError("Loads must be non-negative")
        x0, y0 = get_static_factors(self.bearing_type, contact_angle_deg)
        return max(x0 * radial_load + y0 * axial_load, radial_load)

    def static_safety_factor(self, radial_load, axial_load=0.0, contact_angle_deg=None):
        """Static safety factor s0 = C0 / P0 (ISO 76).

        Values greater than 1 mean the static rating exceeds the static
        equivalent load.  Usual guidance: s0 >= 1.5 for shock loading,
        ~1.0 for normal service, ~0.5 where some brinelling is tolerable.

        Args:
            radial_load (float): Radial load Fr in N.
            axial_load (float): Axial load Fa in N (default: 0).
            contact_angle_deg (float): Contact angle in degrees where the
                bearing family needs it.

        Returns:
            float: Static safety factor s0.

        Raises:
            ValueError: If the bearing has no C0 rating, or the static
                equivalent load works out to zero.
        """
        if self.C0 is None:
            raise ValueError("Static rating requires C0; pass C0= to the constructor")
        p0 = self.static_equivalent_load(radial_load, axial_load, contact_angle_deg)
        if p0 <= 0:
            raise ValueError("Static equivalent load must be strictly positive")
        return self.C0 / p0

    def dn_value(self, speed):
        """Speed factor n*dm in mm*rpm.

        Args:
            speed (float): Operating speed in rpm (or a pint.Quantity).

        Returns:
            float: The n*dm product in mm*rpm.

        Raises:
            ValueError: If the speed is non-positive.
        """
        speed = to_magnitude(speed, "revolutions_per_minute")
        if speed <= 0:
            raise ValueError("Speed must be strictly positive")
        return self.mean_diameter * speed

    def speed_limit(self, lubrication="grease"):
        """Limiting speed in rpm for this bearing size and lubrication.

        Derived from the representative n*dm limits in
        :data:`~mecapy.bearings.iso281_data.LIMITING_SPEED_FACTORS`; a
        first-pass feasibility check, not a substitute for the catalog.

        Args:
            lubrication (str): "grease" (default) or "oil".

        Returns:
            float: Limiting speed in rpm.
        """
        return get_limiting_dn(self.bearing_type, lubrication) / self.mean_diameter

    def speed_check(self, speed, lubrication="grease"):
        """Whether an operating speed is within the limiting speed.

        Args:
            speed (float): Operating speed in rpm (or a pint.Quantity).
            lubrication (str): "grease" (default) or "oil".

        Returns:
            dict: ``speed`` (rpm), ``dn`` (mm*rpm), ``limiting_dn``,
            ``limiting_speed`` (rpm), ``margin`` (limiting/operating) and
            ``within_limit`` (bool).
        """
        speed_rpm = to_magnitude(speed, "revolutions_per_minute")
        dn = self.dn_value(speed_rpm)
        limiting_dn = get_limiting_dn(self.bearing_type, lubrication)
        limiting_speed = limiting_dn / self.mean_diameter
        return {
            "speed": speed_rpm,
            "dn": dn,
            "limiting_dn": limiting_dn,
            "limiting_speed": limiting_speed,
            "margin": limiting_dn / dn,
            "within_limit": dn <= limiting_dn,
        }

    # ---- Tapered roller thrust (Shigley sec. 11-11) ----

    def induced_thrust(self, radial_load, K=TAPERED_K_DEFAULT):
        """Thrust a radial load induces in a tapered roller bearing.

        ::

            Fi = 0.47 * Fr / K                      (Shigley sec. 11-11)

        The taper turns radial load into axial load, which is why tapered
        rollers are mounted in opposed pairs.

        Args:
            radial_load (float): Radial load Fr in N (or a pint.Quantity).
            K (float): Bearing K factor (default: 1.5).

        Returns:
            float: Induced axial load Fi in N.

        Raises:
            ValueError: For a negative load or non-positive K.
        """
        radial_load = to_magnitude(radial_load, "N")
        if radial_load < 0:
            raise ValueError("Loads must be non-negative")
        if K <= 0:
            raise ValueError("K factor must be strictly positive")
        return 0.47 * radial_load / K

    @staticmethod
    def tapered_pair_loads(
        radial_A,
        radial_B,
        external_thrust=0.0,
        K_A=TAPERED_K_DEFAULT,
        K_B=TAPERED_K_DEFAULT,
    ):
        """Equivalent loads of an opposed tapered roller pair.

        Shigley's two-case rule (sec. 11-11): the row whose induced
        thrust the external thrust reinforces carries the combined load,
        the other one carries its radial load alone.  Row A is the row
        the external thrust pushes against.

        ::

            if Fae + 0.47*FrB/KB >= 0.47*FrA/KA:
                FeA = 0.4*FrA + KA*(0.47*FrB/KB + Fae);  FeB = FrB
            else:
                FeB = 0.4*FrB + KB*(0.47*FrA/KA - Fae);  FeA = FrA

        each floored at its own radial load.

        Args:
            radial_A (float): Radial load on row A in N.
            radial_B (float): Radial load on row B in N.
            external_thrust (float): External axial load Fae in N,
                positive towards row A (default: 0).
            K_A (float): K factor of row A (default: 1.5).
            K_B (float): K factor of row B (default: 1.5).

        Returns:
            tuple: ``(Fe_A, Fe_B)`` equivalent radial loads in N.

        Raises:
            ValueError: For negative radial loads or non-positive K.
        """
        radial_A = to_magnitude(radial_A, "N")
        radial_B = to_magnitude(radial_B, "N")
        external_thrust = to_magnitude(external_thrust, "N")
        if radial_A < 0 or radial_B < 0:
            raise ValueError("Loads must be non-negative")
        if K_A <= 0 or K_B <= 0:
            raise ValueError("K factor must be strictly positive")
        induced_A = 0.47 * radial_A / K_A
        induced_B = 0.47 * radial_B / K_B
        if external_thrust + induced_B >= induced_A:
            load_A = 0.4 * radial_A + K_A * (induced_B + external_thrust)
            return max(load_A, radial_A), radial_B
        load_B = 0.4 * radial_B + K_B * (induced_A - external_thrust)
        return radial_A, max(load_B, radial_B)

    # ---- Reports ----

    def duty_cycle_report(self, duty_cycle, application_factor=1.0):
        """Per-segment damage breakdown of a variable-load duty cycle.

        Complements :meth:`equivalent_steady_load` by showing where the
        damage actually comes from: a short severe segment often
        dominates a long mild one.

        Args:
            duty_cycle (list): Segments as accepted by
                :meth:`equivalent_steady_load`.
            application_factor (float): Factor applied to the equivalent
                load when computing the resulting life (default: 1.0).

        Returns:
            dict: ``segments`` (a list of per-segment dicts with
            ``load``, ``fraction``, ``speed``, ``application_factor``,
            ``life`` in revolutions and ``damage_share``),
            ``equivalent_load`` in N, ``life`` in revolutions and
            ``damage_total`` (1.0 by construction when C10 is known).

        Raises:
            ValueError: Per :meth:`equivalent_steady_load`.
        """
        equivalent = self.equivalent_steady_load(duty_cycle)
        a = self.life_exponent
        segments = []
        weighted = []
        for segment in duty_cycle:
            if len(segment) == 2:
                load, fraction = segment
                speed, af = 1.0, 1.0
            elif len(segment) == 3:
                load, fraction, speed = segment
                af = 1.0
            else:
                load, fraction, speed, af = segment
            load = to_magnitude(load, "N")
            speed = to_magnitude(speed, "revolutions_per_minute")
            weighted.append(fraction * speed * (af * load) ** a)
            segments.append(
                {
                    "load": load,
                    "fraction": fraction,
                    "speed": speed,
                    "application_factor": af,
                    "life": self.life(load, af) if self.C10 is not None else None,
                }
            )
        total = sum(weighted)
        for segment, share in zip(segments, weighted):
            segment["damage_share"] = share / total
        return {
            "segments": segments,
            "equivalent_load": equivalent,
            "life": (
                self.life(equivalent, application_factor)
                if self.C10 is not None
                else None
            ),
            "damage_total": sum(segment["damage_share"] for segment in segments),
        }

    def describe(self):
        """
        Human-readable summary of the bearing's geometry and ratings.

        The string is returned, not printed; use ``print(bearing.describe())``.

        Returns:
            str: Multi-line description, one quantity per line as
            ``label (symbol) = value unit``.
        """
        header = f"{self.__class__.__name__} geometry"
        if self.name:
            header += f" '{self.name}'"
        lines = [
            header,
            "=" * 40,
            f"bore diameter (d) = {self.bore_diameter:.3f} mm",
            f"outer diameter (D) = {self.outer_diameter:.3f} mm",
            f"width (B) = {self.width:.3f} mm",
            f"mean diameter (dm) = {self.mean_diameter:.3f} mm",
            f"bearing type = {self.bearing_type}",
            f"life exponent (a) = {self.life_exponent:.4f}",
            f"rating life (L10) = {self.rating_life:.4g} rev",
        ]
        if self.C10 is not None:
            lines.append(f"dynamic rating (C10) = {self.C10:.1f} N")
        else:
            lines.append("dynamic rating (C10) = not given")
        if self.C0 is not None:
            lines.append(f"static rating (C0) = {self.C0:.1f} N")
        else:
            lines.append("static rating (C0) = not given")
        lines.append(f"material = {self.material}")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"Bearing({self.bore_diameter}/{self.outer_diameter}x{self.width}, "
            f"type={self.bearing_type})"
        )
