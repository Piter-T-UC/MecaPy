"""Data tables for rolling-contact bearing rating (Shigley ch. 11).

Units: forces in N, life in revolutions unless noted otherwise.

Tables digitized from Shigley's Mechanical Engineering Design:
    - Weibull distribution parameters (sec. 11-4)
    - Equivalent radial load factors X and Y (table 11-1)
    - Load-application factors (table 11-5)
"""

import math

import numpy as np

#: Weibull parameters for bearing life distribution (Shigley sec. 11-4).
#: Life is expressed as a dimensionless multiple x of the rating life L10.
WEIBULL_X0 = 0.02  # guaranteed (minimum) life
WEIBULL_THETA = 4.459  # characteristic parameter
WEIBULL_B = 1.483  # shape parameter

#: Life exponent a (eq. 11-1): 3 for ball bearings, 10/3 for roller bearings.
LIFE_EXPONENTS = {
    "ball": 3.0,
    "angular_contact": 3.0,
    "roller": 10.0 / 3.0,
    "cylindrical": 10.0 / 3.0,
    "tapered": 10.0 / 3.0,
}

#: Rating life associated with the catalog load rating C10 (revolutions).
#: Most manufacturers rate at one million revolutions.
DEFAULT_RATING_LIFE = 1.0e6

#: Rotation factor V (table 11-1): which ring rotates relative to the load.
ROTATION_FACTORS = {"inner": 1.0, "outer": 1.2}

#: Equivalent radial load factors for single-row ball bearings
#: (Shigley table 11-1).  Rows are indexed by Fa/C0; the factors are
#: X1 = 1, Y1 = 0 when Fa/(V*Fr) <= e and X2 = 0.56, Y2 below otherwise.
XY_FA_OVER_C0 = [
    0.014,
    0.021,
    0.028,
    0.042,
    0.056,
    0.070,
    0.084,
    0.110,
    0.17,
    0.28,
    0.42,
    0.56,
]
XY_E = [0.19, 0.21, 0.22, 0.24, 0.26, 0.27, 0.28, 0.30, 0.34, 0.38, 0.42, 0.44]
XY_Y2 = [2.30, 2.15, 1.99, 1.85, 1.71, 1.63, 1.55, 1.45, 1.31, 1.15, 1.04, 1.00]
XY_X1 = 1.00
XY_Y1 = 0.00
XY_X2 = 0.56

#: Bearing types the table 11-1 X/Y factors above may be applied to.
#: Membership is the gate for combined (radial + axial) loading: roller
#: families under thrust need manufacturer data.  This is deliberately a
#: separate set from :data:`LIFE_EXPONENTS` -- a future ball-family type
#: would share the exponent a = 3 without necessarily sharing these X/Y
#: factors, so the load model must not infer one from the other.
XY_TABLE_TYPES = frozenset({"ball", "angular_contact"})

#: Load-application factors (Shigley table 11-5), as (min, max) ranges.
APPLICATION_FACTORS = {
    "precision gearing": (1.0, 1.1),
    "commercial gearing": (1.1, 1.3),
    "poor bearing seals": (1.2, 1.2),
    "no impact": (1.0, 1.2),
    "light impact": (1.2, 1.5),
    "moderate impact": (1.5, 3.0),
}


def get_life_exponent(bearing_type):
    """Return the load-life exponent a for a bearing type (eq. 11-1).

    Args:
        bearing_type (str): One of the keys of ``LIFE_EXPONENTS``.

    Returns:
        float: Exponent a (3 for ball, 10/3 for roller families).

    Raises:
        ValueError: If the bearing type is unknown.
    """
    if bearing_type not in LIFE_EXPONENTS:
        raise ValueError(
            f"Unknown bearing type '{bearing_type}'. "
            f"Available: {sorted(LIFE_EXPONENTS)}"
        )
    return LIFE_EXPONENTS[bearing_type]


def get_xy_factors(fa_over_c0):
    """Interpolate the table 11-1 factors for a given Fa/C0 ratio.

    Values of e and Y2 are linearly interpolated on Fa/C0; outside the
    tabulated range (0.014 to 0.56) the end rows are used, matching
    Shigley's practice (the table itself is bounded).

    Args:
        fa_over_c0 (float): Ratio of axial load to static rating, Fa/C0.

    Returns:
        tuple: (e, x2, y2) where e is the load-ratio threshold and
        (x2, y2) apply when Fa/(V*Fr) > e.  X1 = 1, Y1 = 0 otherwise.

    Raises:
        ValueError: If fa_over_c0 is negative.
    """
    if fa_over_c0 < 0:
        raise ValueError("Fa/C0 must be non-negative")
    e = float(np.interp(fa_over_c0, XY_FA_OVER_C0, XY_E))
    y2 = float(np.interp(fa_over_c0, XY_FA_OVER_C0, XY_Y2))
    return e, XY_X2, y2


def get_application_factor(service, level="mid"):
    """Look up a load-application factor af (table 11-5).

    Args:
        service (str): Service description, one of the keys of
            ``APPLICATION_FACTORS`` (e.g. "commercial gearing").
        level (str): "min", "mid" or "max" of the tabulated range
            (default: "mid").

    Returns:
        float: Application factor af (>= 1).

    Raises:
        ValueError: If the service or level is unknown.
    """
    if service not in APPLICATION_FACTORS:
        raise ValueError(
            f"Unknown service '{service}'. " f"Available: {sorted(APPLICATION_FACTORS)}"
        )
    lo, hi = APPLICATION_FACTORS[service]
    if level == "min":
        return lo
    if level == "max":
        return hi
    if level == "mid":
        return (lo + hi) / 2.0
    raise ValueError("level must be 'min', 'mid' or 'max'")


def weibull_reliability(x):
    """Reliability at a dimensionless life x (Shigley eq. 11-5).

    R = exp(-((x - x0) / (theta - x0))**b) for x >= x0, else 1.0,
    where x is the life expressed as a multiple of the rating life L10.

    Args:
        x (float): Life as a multiple of L10 (must be non-negative).

    Returns:
        float: Probability of survival, 0 < R <= 1.

    Raises:
        ValueError: If x is negative.
    """
    if x < 0:
        raise ValueError("Life multiple x must be non-negative")
    if x <= WEIBULL_X0:
        return 1.0
    return math.exp(-(((x - WEIBULL_X0) / (WEIBULL_THETA - WEIBULL_X0)) ** WEIBULL_B))


def weibull_life_multiplier(reliability):
    """Dimensionless life x achieving a given reliability (eq. 11-5 inverted).

    x = x0 + (theta - x0) * (ln(1/R))**(1/b).  At R = 0.90 this returns
    approximately 0.99, i.e. the Weibull fit reproduces the rating life L10.

    Args:
        reliability (float): Desired probability of survival, 0 < R < 1.

    Returns:
        float: Life as a multiple of L10.

    Raises:
        ValueError: If reliability is not strictly between 0 and 1.
    """
    if not 0.0 < reliability < 1.0:
        raise ValueError("Reliability must be strictly between 0 and 1")
    z = math.log(1.0 / reliability) ** (1.0 / WEIBULL_B)
    return WEIBULL_X0 + (WEIBULL_THETA - WEIBULL_X0) * z
