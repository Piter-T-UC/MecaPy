"""ISO 281 / ISO 76 rolling-bearing rating data (modified rating life).

Provenance: ISO 281:2007 (modified rating life L_nm, life modification
factor a_ISO, reliability factor a1, contamination factor e_C, reference
viscosity nu1) and ISO 76:2006 (static equivalent load factors X0/Y0).
The a_ISO closed forms are the ISO 281 Annex expressions as commonly
reproduced by bearing manufacturers; every coefficient block below is
marked ``confirmed`` (cross-checked against the published form) or
``ESTIMATED`` (representative value, verify against a catalog before
relying on it for a real selection).

This module is deliberately separate from :mod:`mecapy.bearings.bearing_data`,
which carries Shigley-ch.-11 provenance.  The two life models coexist:
``Bearing.adjusted_life`` follows Shigley's 3-parameter Weibull, while
``Bearing.iso_life`` follows ISO 281.  They answer the same question with
different conventions and will not agree numerically.

Units: diameters in mm, speeds in rpm, kinematic viscosity in mm^2/s
(= cSt), loads in N.  All factors are dimensionless.

Fields per contamination row:
    e_c_small (tuple): (min, max) e_C for a mean diameter below 100 mm.
    e_c_large (tuple): (min, max) e_C for a mean diameter of 100 mm or more.

Fields per static-factor row:
    X0 (float): Radial factor of the static equivalent load.
    Y0 (float or None): Axial factor; None means it follows from the
        contact angle as 0.22 * cot(alpha) and must be supplied.

Fields per limiting-speed row:
    grease (float): Limiting n*dm product in mm*rpm for grease lubrication.
    oil (float): Limiting n*dm product in mm*rpm for oil lubrication.
"""

import math

import numpy as np

#: Upper and lower clamps on the life modification factor (ISO 281 caps
#: a_ISO at 50; below 0.1 the model is outside its validated range).
A_ISO_MAX = 50.0
A_ISO_MIN = 0.1

#: Valid band of the viscosity ratio kappa = nu / nu1 (ISO 281).  Outside
#: it the a_ISO curves are not defined and the ends are used instead.
KAPPA_MIN = 0.1
KAPPA_MAX = 4.0

#: Default lubricant density in kg/m^3, used to convert the dynamic
#: viscosity of :mod:`mecapy.bearings.lubrication_data` (mPa*s) into the
#: kinematic viscosity (mm^2/s) ISO 281 works in.  Typical mineral oil.
DEFAULT_OIL_DENSITY = 870.0  # kg/m^3 (confirmed: typical mineral oil)

#: Ratio C0/Cu used to estimate the fatigue load limit when a catalog Cu
#: is unavailable.  A rule of thumb, not an ISO value -- an explicit Cu
#: always wins.
FATIGUE_LOAD_LIMIT_RATIO = 8.2  # ESTIMATED (common manufacturer rule)

#: ISO 281 reliability factor a1 for the modified rating life L_nm.
#: Deliberately NOT the Shigley Weibull multiplier in bearing_data.py.
RELIABILITY_FACTORS_A1 = {
    0.90: 1.0,  # L10 (confirmed: ISO 281:2007 Table 1)
    0.95: 0.64,  # L5   (confirmed)
    0.96: 0.55,  # L4   (confirmed)
    0.97: 0.47,  # L3   (confirmed)
    0.98: 0.37,  # L2   (confirmed)
    0.99: 0.25,  # L1   (confirmed)
    0.995: 0.175,  # L0.5 (confirmed)
    0.996: 0.16,  # L0.4 (confirmed)
    0.997: 0.145,  # L0.3 (confirmed)
    0.998: 0.12,  # L0.2 (confirmed)
    0.999: 0.093,  # L0.1 (confirmed)
}

#: Contamination factor e_C (ISO 281:2007 Table 13), as (min, max) bands
#: either side of a mean diameter of 100 mm.  Cleanliness is by far the
#: most influential input to a_ISO -- a severely contaminated bearing can
#: lose an order of magnitude of life at the same load.
CONTAMINATION_LEVELS = {
    "extreme_cleanliness": {
        "e_c_small": (1.0, 1.0),  # confirmed
        "e_c_large": (1.0, 1.0),  # confirmed
    },
    "high_cleanliness": {
        "e_c_small": (0.6, 0.8),  # confirmed
        "e_c_large": (0.8, 0.9),  # confirmed
    },
    "normal_cleanliness": {
        "e_c_small": (0.5, 0.6),  # confirmed
        "e_c_large": (0.6, 0.8),  # confirmed
    },
    "slight_contamination": {
        "e_c_small": (0.3, 0.5),  # confirmed
        "e_c_large": (0.4, 0.6),  # confirmed
    },
    "typical_contamination": {
        "e_c_small": (0.1, 0.3),  # confirmed
        "e_c_large": (0.2, 0.4),  # confirmed
    },
    "severe_contamination": {
        "e_c_small": (0.0, 0.1),  # confirmed
        "e_c_large": (0.0, 0.1),  # confirmed
    },
    "very_severe_contamination": {
        "e_c_small": (0.0, 0.0),  # confirmed
        "e_c_large": (0.0, 0.0),  # confirmed
    },
}

#: Coefficients of the ISO 281 a_ISO closed form, per bearing family and
#: kappa band.  The expression is
#:
#:     a_ISO = 0.1 * [1 - (c1 - c2 / kappa**c3)**exponent
#:                        * (e_C * Cu / P)**power] ** (-slope)
#:
#: with the band selected by kappa.  Only the radial families are
#: implemented: the thrust forms use different constants and are rejected
#: rather than silently approximated by the radial ones.
ISO281_A_ISO_COEFFICIENTS = {
    "radial_ball": {
        "exponent": 0.83,  # confirmed: ISO 281:2007 Annex, radial ball
        "power": 1.0 / 3.0,  # confirmed
        "slope": 9.3,  # confirmed
        "bands": [
            # (kappa_low, kappa_high, c1, c2, c3)
            (0.1, 0.4, 2.5671, 2.2649, 0.054381),  # confirmed
            (0.4, 1.0, 2.5671, 1.9987, 0.019087),  # confirmed
            (1.0, 4.0, 2.5671, 1.9987, 0.071739),  # confirmed
        ],
    },
    "radial_roller": {
        "exponent": 0.4,  # confirmed: ISO 281:2007 Annex, radial roller
        "power": 0.4,  # confirmed
        "slope": 9.185,  # confirmed
        "bands": [
            (0.1, 0.4, 1.5859, 1.3993, 0.054381),  # confirmed
            (0.4, 1.0, 1.5859, 1.2348, 0.19087),  # confirmed
            (1.0, 4.0, 1.5859, 1.2348, 0.071739),  # confirmed
        ],
    },
}

#: Which a_ISO family each bearing type belongs to.
A_ISO_FAMILIES = {
    "ball": "radial_ball",
    "angular_contact": "radial_ball",
    "roller": "radial_roller",
    "cylindrical": "radial_roller",
    "tapered": "radial_roller",
}

#: Static equivalent load factors (ISO 76:2006).  A ``Y0`` of None means
#: the factor depends on the contact angle: Y0 = 0.22 * cot(alpha).
STATIC_LOAD_FACTORS = {
    "ball": {"X0": 0.6, "Y0": 0.5},  # confirmed: single-row deep groove
    "angular_contact": {"X0": 0.5, "Y0": None},  # confirmed: 0.22*cot(alpha)
    "roller": {"X0": 1.0, "Y0": 0.0},  # confirmed: radial roller, no thrust
    "cylindrical": {"X0": 1.0, "Y0": 0.0},  # confirmed
    "tapered": {"X0": 0.5, "Y0": None},  # confirmed: 0.22*cot(alpha)
}

#: Limiting speed as an n*dm product in mm*rpm, per family and lubrication.
#: Representative catalog magnitudes for a first-pass feasibility check,
#: not a substitute for the specific bearing's catalog page.
LIMITING_SPEED_FACTORS = {
    "ball": {"grease": 300000.0, "oil": 500000.0},  # ESTIMATED
    "angular_contact": {"grease": 250000.0, "oil": 400000.0},  # ESTIMATED
    "roller": {"grease": 250000.0, "oil": 350000.0},  # ESTIMATED
    "cylindrical": {"grease": 300000.0, "oil": 400000.0},  # ESTIMATED
    "tapered": {"grease": 200000.0, "oil": 300000.0},  # ESTIMATED
}

#: Default K factor of a tapered roller bearing (Shigley sec. 11-11),
#: used by the induced-thrust relation Fi = 0.47 * Fr / K.
TAPERED_K_DEFAULT = 1.5  # confirmed: Shigley sec. 11-11


def reference_viscosity(mean_diameter, speed_rpm):
    """Reference kinematic viscosity nu1 for adequate film formation.

    Closed-form fit of the ISO 281 nu1 chart, so no chart digitization is
    involved::

        nu1 = 45000 * n**-0.83 * dm**-0.5     n < 1000 rpm
        nu1 =  4500 * n**-0.5  * dm**-0.5     n >= 1000 rpm

    Args:
        mean_diameter (float): Bearing mean diameter dm = (d + D)/2 in mm.
        speed_rpm (float): Operating speed in rpm.

    Returns:
        float: Reference kinematic viscosity nu1 in mm^2/s.

    Raises:
        ValueError: If the diameter or speed is not strictly positive.
    """
    if mean_diameter <= 0:
        raise ValueError("Mean diameter must be strictly positive")
    if speed_rpm <= 0:
        raise ValueError("Speed must be strictly positive")
    if speed_rpm < 1000.0:
        return 45000.0 * speed_rpm**-0.83 * mean_diameter**-0.5
    return 4500.0 * speed_rpm**-0.5 * mean_diameter**-0.5


def viscosity_ratio(kinematic_viscosity, mean_diameter, speed_rpm):
    """Viscosity ratio kappa = nu / nu1 (ISO 281).

    kappa below 1 means the film is thinner than the surface roughness
    can tolerate unaided; kappa of 4 is the practical ceiling beyond
    which further viscosity buys no life.

    Args:
        kinematic_viscosity (float): Operating viscosity nu in mm^2/s.
        mean_diameter (float): Mean diameter dm = (d + D)/2 in mm.
        speed_rpm (float): Operating speed in rpm.

    Returns:
        float: Viscosity ratio kappa (unclamped; :func:`a_iso` clamps it
        to the validated band 0.1 to 4).

    Raises:
        ValueError: If the viscosity, diameter or speed is not strictly
            positive.
    """
    if kinematic_viscosity <= 0:
        raise ValueError("Kinematic viscosity must be strictly positive")
    return kinematic_viscosity / reference_viscosity(mean_diameter, speed_rpm)


def kinematic_from_dynamic(dynamic_viscosity_mpas, density=DEFAULT_OIL_DENSITY):
    """Convert dynamic viscosity (mPa*s) to kinematic viscosity (mm^2/s).

    Bridges :func:`mecapy.bearings.lubrication_data.viscosity`, which is
    the ch.-12 journal-bearing convention (mPa*s), into the ISO 281
    convention (mm^2/s): nu = mu / rho with mu in Pa*s and rho in kg/m^3
    gives m^2/s, hence the 1e6 factor to mm^2/s.

    Args:
        dynamic_viscosity_mpas (float): Absolute viscosity in mPa*s (= cP).
        density (float): Lubricant density in kg/m^3 (default: 870).

    Returns:
        float: Kinematic viscosity in mm^2/s (= cSt).

    Raises:
        ValueError: If the viscosity or density is not strictly positive.
    """
    if dynamic_viscosity_mpas <= 0:
        raise ValueError("Dynamic viscosity must be strictly positive")
    if density <= 0:
        raise ValueError("Density must be strictly positive")
    return dynamic_viscosity_mpas * 1e-3 / density * 1e6


def get_reliability_factor(reliability):
    """ISO 281 reliability factor a1 for the modified rating life.

    Interpolates linearly between the tabulated survival probabilities.

    Args:
        reliability (float): Probability of survival, 0.90 <= R <= 0.999.

    Returns:
        float: Reliability factor a1 (1.0 at R = 0.90).

    Raises:
        ValueError: If the reliability is outside the tabulated range.
    """
    grid = sorted(RELIABILITY_FACTORS_A1)
    if not grid[0] <= reliability <= grid[-1]:
        raise ValueError(
            f"Reliability {reliability!r} is outside the ISO 281 table "
            f"({grid[0]} to {grid[-1]})"
        )
    values = [RELIABILITY_FACTORS_A1[key] for key in grid]
    return float(np.interp(reliability, grid, values))


def get_contamination_factor(level, mean_diameter, position="mid"):
    """Contamination factor e_C (ISO 281 Table 13).

    Args:
        level (str): Cleanliness level, one of the keys of
            :data:`CONTAMINATION_LEVELS` (e.g. "normal_cleanliness").
        mean_diameter (float): Mean diameter dm in mm; the table splits
            at 100 mm.
        position (str): Which end of the tabulated band to take, "min",
            "mid" (default) or "max".

    Returns:
        float: Contamination factor e_C, 0 to 1.

    Raises:
        ValueError: If the level, position or diameter is invalid.
    """
    if level not in CONTAMINATION_LEVELS:
        available = ", ".join(sorted(CONTAMINATION_LEVELS))
        raise ValueError(
            f"Unknown contamination level {level!r}. Available: {available}"
        )
    if mean_diameter <= 0:
        raise ValueError("Mean diameter must be strictly positive")
    if position not in ("min", "mid", "max"):
        raise ValueError("position must be 'min', 'mid' or 'max'")
    key = "e_c_small" if mean_diameter < 100.0 else "e_c_large"
    low, high = CONTAMINATION_LEVELS[level][key]
    if position == "min":
        return low
    if position == "max":
        return high
    return 0.5 * (low + high)


def get_static_factors(bearing_type, contact_angle_deg=None):
    """Static equivalent load factors X0, Y0 (ISO 76).

    Args:
        bearing_type (str): Bearing type, one of the keys of
            :data:`STATIC_LOAD_FACTORS`.
        contact_angle_deg (float): Contact angle alpha in degrees,
            required for the angular-contact and tapered families where
            Y0 = 0.22 * cot(alpha).

    Returns:
        tuple: ``(X0, Y0)``, both dimensionless.

    Raises:
        ValueError: If the type is unknown, or the contact angle is
            missing or outside (0, 90) for a type that needs it.
    """
    if bearing_type not in STATIC_LOAD_FACTORS:
        available = ", ".join(sorted(STATIC_LOAD_FACTORS))
        raise ValueError(
            f"Unknown bearing type {bearing_type!r}. Available: {available}"
        )
    row = STATIC_LOAD_FACTORS[bearing_type]
    if row["Y0"] is not None:
        return row["X0"], row["Y0"]
    if contact_angle_deg is None:
        raise ValueError(
            f"Bearing type {bearing_type!r} needs contact_angle_deg "
            "(Y0 = 0.22 * cot(alpha))"
        )
    if not 0.0 < contact_angle_deg < 90.0:
        raise ValueError("Contact angle must be in (0, 90) degrees")
    return row["X0"], 0.22 / math.tan(math.radians(contact_angle_deg))


def get_limiting_dn(bearing_type, lubrication="grease"):
    """Limiting speed factor n*dm in mm*rpm.

    Args:
        bearing_type (str): Bearing type, one of the keys of
            :data:`LIMITING_SPEED_FACTORS`.
        lubrication (str): "grease" (default) or "oil".

    Returns:
        float: Limiting n*dm product in mm*rpm.

    Raises:
        ValueError: If the type or lubrication is unknown.
    """
    if bearing_type not in LIMITING_SPEED_FACTORS:
        available = ", ".join(sorted(LIMITING_SPEED_FACTORS))
        raise ValueError(
            f"Unknown bearing type {bearing_type!r}. Available: {available}"
        )
    row = LIMITING_SPEED_FACTORS[bearing_type]
    if lubrication not in row:
        available = ", ".join(sorted(row))
        raise ValueError(f"Unknown lubrication {lubrication!r}. Available: {available}")
    return row[lubrication]


def a_iso(load_ratio, kappa, family="radial_ball"):
    """Life modification factor a_ISO (ISO 281:2007).

    ::

        a_ISO = 0.1 * [1 - (c1 - c2 / kappa^c3)^e * (eC*Cu/P)^p]^(-s)
                                                    (ISO 281 Annex)

    with (c1, c2, c3) selected by the kappa band and (e, p, s) by the
    bearing family.  The result is clamped to [0.1, 50]: the standard
    caps a_ISO at 50, and the bracket goes non-positive for very dirty,
    lightly loaded cases where the model no longer applies.

    Args:
        load_ratio (float): The product ``e_C * Cu / P`` (dimensionless),
            where Cu is the fatigue load limit and P the equivalent load.
        kappa (float): Viscosity ratio nu/nu1; clamped to [0.1, 4].
        family (str): "radial_ball" or "radial_roller".

    Returns:
        float: Life modification factor a_ISO, between 0.1 and 50.

    Raises:
        ValueError: If the load ratio is negative or the family unknown.
    """
    if load_ratio < 0:
        raise ValueError("Load ratio eC*Cu/P must be non-negative")
    if family not in ISO281_A_ISO_COEFFICIENTS:
        available = ", ".join(sorted(ISO281_A_ISO_COEFFICIENTS))
        raise ValueError(f"Unknown a_ISO family {family!r}. Available: {available}")
    if load_ratio == 0.0:
        return A_ISO_MIN
    kappa = min(max(kappa, KAPPA_MIN), KAPPA_MAX)
    row = ISO281_A_ISO_COEFFICIENTS[family]
    for low, high, c1, c2, c3 in row["bands"]:
        if low <= kappa <= high:
            break
    bracket = (
        1.0 - (c1 - c2 / kappa**c3) ** row["exponent"] * load_ratio ** row["power"]
    )
    if bracket <= 0.0:
        return A_ISO_MAX
    value = 0.1 * bracket ** -row["slope"]
    return min(max(value, A_ISO_MIN), A_ISO_MAX)


def get_a_iso_family(bearing_type):
    """a_ISO family a bearing type belongs to.

    Args:
        bearing_type (str): Bearing type, one of the keys of
            :data:`A_ISO_FAMILIES`.

    Returns:
        str: "radial_ball" or "radial_roller".

    Raises:
        ValueError: If the bearing type has no ISO 281 radial family.
    """
    if bearing_type not in A_ISO_FAMILIES:
        available = ", ".join(sorted(A_ISO_FAMILIES))
        raise ValueError(
            f"Unknown bearing type {bearing_type!r}. Available: {available}"
        )
    return A_ISO_FAMILIES[bearing_type]
