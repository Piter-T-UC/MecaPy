"""Rolling-contact bearing design and analysis module (Shigley ch. 11).

Units: dimensions in mm, forces and load ratings in N, life in revolutions
unless noted otherwise.  Catalog ratings quoted in kN must be multiplied
by 1000 before being passed in.
"""

import math

from ..base import MechaElement
from .bearing_data import (
    DEFAULT_RATING_LIFE,
    ROTATION_FACTORS,
    WEIBULL_B,
    WEIBULL_THETA,
    WEIBULL_X0,
    get_life_exponent,
    get_xy_factors,
    weibull_life_multiplier,
    weibull_reliability,
)


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
            bore_diameter (float): Bearing bore diameter in mm.
            outer_diameter (float): Bearing outer diameter in mm.
            width (float): Bearing width in mm.
            bearing_type (str): Type of bearing (default: "ball").
                One of "ball", "angular_contact", "roller", "cylindrical",
                "tapered"; sets the load-life exponent (3 or 10/3).
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the bearing.
            C10 (float): Basic dynamic load rating in N.  Catalog values
                in kN must be multiplied by 1000.  Optional; life methods
                require it.
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
        if bore_diameter <= 0:
            raise ValueError("Bore diameter must be strictly positive")
        if width <= 0:
            raise ValueError("Width must be strictly positive")
        if outer_diameter <= bore_diameter:
            raise ValueError("Outer diameter must be larger than bore diameter")
        if C10 is not None and C10 <= 0:
            raise ValueError("Dynamic load rating C10 must be strictly positive")
        if C0 is not None and C0 <= 0:
            raise ValueError("Static load rating C0 must be strictly positive")
        if rating_life <= 0:
            raise ValueError("Rating life must be strictly positive")
        self.bore_diameter = bore_diameter
        self.outer_diameter = outer_diameter
        self.width = width
        self.bearing_type = bearing_type
        self.life_exponent = get_life_exponent(bearing_type)
        self.C10 = C10
        self.C0 = C0
        self.rating_life = rating_life

    def _require_c10(self):
        if self.C10 is None:
            raise ValueError("Bearing has no C10 rating; pass C10= to the constructor")

    def life(self, load, application_factor=1.0):
        """Rating (L10) life at a steady radial load (eq. 11-1/11-3).

        L = rating_life * (C10 / (af * load))**a, with a = 3 for ball
        and 10/3 for roller bearings.

        Args:
            load (float): Steady equivalent radial load in N.
            application_factor (float): Load-application factor af
                (table 11-5, default: 1.0).

        Returns:
            float: Life in revolutions with 90% reliability.

        Raises:
            ValueError: If the load or factor is non-positive, or the
                bearing has no C10 rating.
        """
        self._require_c10()
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
            load (float): Steady equivalent radial load in N.
            speed (float): Shaft speed in rpm.
            application_factor (float): Load-application factor af
                (default: 1.0).

        Returns:
            float: Life in hours with 90% reliability.

        Raises:
            ValueError: If the speed is non-positive (plus the checks
                of :meth:`life`).
        """
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
            load (float): Design radial load in N.
            desired_life (float): Required life in revolutions.
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
            radial_load (float): Radial load Fr in N.
            axial_load (float): Axial (thrust) load Fa in N (default: 0).
            rotating (str): Which ring rotates, "inner" or "outer"
                (sets V = 1.0 or 1.2, default: "inner").

        Returns:
            float: Equivalent steady radial load P in N.

        Raises:
            ValueError: For negative loads, an unknown rotating ring, a
                roller-family bearing with axial load, or a missing C0
                rating when axial load is present.
        """
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
        if self.life_exponent != 3.0:
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
                Omitted speeds weight all segments equally.

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
            if load <= 0 or fraction <= 0 or speed <= 0 or af <= 0:
                raise ValueError(
                    "Loads, fractions, speeds and factors must be strictly positive"
                )
            weight = fraction * speed
            numerator += weight * (af * load) ** a
            denominator += weight
        return (numerator / denominator) ** (1.0 / a)

    def __repr__(self):
        return (
            f"Bearing({self.bore_diameter}/{self.outer_diameter}x{self.width}, "
            f"type={self.bearing_type})"
        )
