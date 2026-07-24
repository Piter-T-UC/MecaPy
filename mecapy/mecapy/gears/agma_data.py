"""Data tables for AGMA gear rating (Shigley ch. 14 as reference).

Contains the Lewis form factors, digitized AGMA bending geometry factor
(J / YJ) grids, load-distribution coefficients, reliability factors,
allowable-stress fits for gear steels and worm wear factors.

The J grids are digitized from Shigley's Figures 14-6 (spur), 14-7 and
14-8 (helical); figure-reading accuracy is a few percent, and any rating
accepts an explicit ``YJ=`` override.

Units: metric (mm, MPa) unless noted; face width enters the Cma fit in
inches (converted by the caller).
"""

import math

import numpy as np

# ----------------------------------------------------------------------
# Lewis form factor Y — Shigley Table 14-2 (20 deg full-depth teeth)
# ----------------------------------------------------------------------

#: Lewis form factor Y vs tooth count (load near middle of tooth height).
LEWIS_Y = {
    12: 0.245, 13: 0.261, 14: 0.277, 15: 0.290, 16: 0.296,
    17: 0.303, 18: 0.309, 19: 0.314, 20: 0.322, 21: 0.328,
    22: 0.331, 24: 0.337, 26: 0.346, 28: 0.353, 30: 0.359,
    34: 0.371, 38: 0.384, 43: 0.397, 50: 0.409, 60: 0.422,
    75: 0.435, 100: 0.447, 150: 0.460, 300: 0.472, 400: 0.480,
}

#: Lewis form factor for a rack (infinite tooth count).
LEWIS_Y_RACK = 0.485


def lewis_form_factor(teeth):
    """
    Lewis form factor Y for 20-degree full-depth teeth.

    Log-interpolates Shigley Table 14-2. Tooth counts above 400 use the
    rack value.

    Args:
        teeth (float): Number of teeth (may be fractional, e.g. virtual
            bevel teeth). Must be >= 12.

    Returns:
        float: Lewis form factor Y (dimensionless).

    Raises:
        ValueError: If ``teeth`` is below the table range (12).
    """
    if teeth < 12:
        raise ValueError("Lewis table starts at 12 teeth")
    if teeth >= 400:
        return LEWIS_Y_RACK
    counts = sorted(LEWIS_Y)
    values = [LEWIS_Y[n] for n in counts]
    return float(np.interp(math.log(teeth),
                           [math.log(n) for n in counts], values))


# ----------------------------------------------------------------------
# AGMA bending geometry factor J (YJ) — digitized from Shigley Fig. 14-6
# (spur, 20 deg full-depth) and Figs. 14-7 / 14-8 (helical).
# ----------------------------------------------------------------------

#: Rated-gear tooth counts (rows of SPUR_J).
SPUR_J_TEETH = [12, 15, 17, 21, 26, 35, 55, 135, 400]

#: Mating-gear tooth counts (columns of SPUR_J); 1000 stands for a rack.
SPUR_J_MATING = [17, 25, 35, 50, 85, 170, 1000]

#: J values, rows = rated gear teeth, columns = mating gear teeth.
SPUR_J = [
    # mate:  17     25     35     50     85     170    1000/rack
    [0.210, 0.215, 0.220, 0.225, 0.230, 0.235, 0.240],  # 12
    [0.245, 0.250, 0.255, 0.260, 0.265, 0.270, 0.275],  # 15
    [0.290, 0.295, 0.298, 0.300, 0.305, 0.310, 0.315],  # 17
    [0.320, 0.325, 0.330, 0.335, 0.340, 0.345, 0.350],  # 21
    [0.345, 0.350, 0.355, 0.360, 0.365, 0.370, 0.375],  # 26
    [0.373, 0.380, 0.385, 0.390, 0.395, 0.400, 0.405],  # 35
    [0.405, 0.410, 0.415, 0.420, 0.425, 0.430, 0.435],  # 55
    [0.430, 0.440, 0.445, 0.450, 0.460, 0.465, 0.470],  # 135
    [0.445, 0.455, 0.460, 0.465, 0.475, 0.480, 0.485],  # 400
]

#: Helix angles (deg) for the helical J grids (columns).
HELICAL_J_ANGLES = [5, 10, 15, 20, 25, 30, 35]

#: Rated-gear tooth counts for HELICAL_J75 (rows).
HELICAL_J_TEETH = [20, 30, 60, 150, 500]

#: J' for a helical gear mating with a 75-tooth gear (Fig. 14-7).
HELICAL_J75 = [
    # helix:  5     10     15     20     25     30     35
    [0.440, 0.450, 0.460, 0.460, 0.450, 0.440, 0.410],  # 20
    [0.470, 0.480, 0.490, 0.490, 0.480, 0.460, 0.440],  # 30
    [0.510, 0.520, 0.530, 0.530, 0.520, 0.500, 0.470],  # 60
    [0.540, 0.550, 0.560, 0.560, 0.550, 0.530, 0.500],  # 150
    [0.560, 0.570, 0.580, 0.580, 0.570, 0.550, 0.520],  # 500
]

#: Mating-gear tooth counts for the J-factor multiplier (rows, Fig. 14-8).
HELICAL_JMULT_TEETH = [20, 30, 50, 75, 150, 500]

#: Multiplier applied to HELICAL_J75 for mating gears other than 75 teeth.
HELICAL_J_MULTIPLIER = [
    # helix:  5      10     15     20     25     30     35
    [0.955, 0.955, 0.950, 0.950, 0.945, 0.940, 0.935],  # 20
    [0.970, 0.970, 0.965, 0.965, 0.960, 0.955, 0.955],  # 30
    [0.985, 0.985, 0.985, 0.985, 0.980, 0.980, 0.980],  # 50
    [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 75
    [1.015, 1.015, 1.015, 1.015, 1.020, 1.020, 1.020],  # 150
    [1.030, 1.030, 1.030, 1.030, 1.035, 1.040, 1.045],  # 500
]


def _interp_grid(x, y, x_grid, y_grid, values, log_axes=(False, False)):
    """Bilinear interpolation on a small grid, optionally log-scaled."""
    fx = math.log if log_axes[0] else float
    fy = math.log if log_axes[1] else float
    x = min(max(x, x_grid[0]), x_grid[-1])
    y = min(max(y, y_grid[0]), y_grid[-1])
    xg = [fx(v) for v in x_grid]
    yg = [fy(v) for v in y_grid]
    arr = np.asarray(values, dtype=float)
    # Interpolate along columns first (y), then along rows (x).
    col = [np.interp(fy(y), yg, arr[i, :]) for i in range(len(x_grid))]
    return float(np.interp(fx(x), xg, col))


def geometry_factor_J(teeth, mating_teeth, helix_angle=0.0):
    """
    AGMA bending-strength geometry factor J (YJ), 20 deg pressure angle.

    Spur gears interpolate the digitized Fig. 14-6 grid; helical gears
    use the Fig. 14-7 value (75-tooth mate) times the Fig. 14-8
    mating-teeth multiplier. Values carry figure-reading uncertainty of
    a few percent — pass an explicit ``YJ`` to the rating to override.

    Args:
        teeth (int): Teeth of the gear being rated (>= 12).
        mating_teeth (int or str): Teeth of the mating gear, or "rack".
        helix_angle (float): Helix angle in degrees (0 for spur).

    Returns:
        float: Geometry factor J (dimensionless).

    Raises:
        ValueError: If ``teeth`` is below 12 (undercut region not
            covered by the charts).
    """
    if teeth < 12:
        raise ValueError("J charts start at 12 teeth")
    if mating_teeth == "rack":
        mating_teeth = 1000
    if helix_angle == 0:
        return _interp_grid(teeth, mating_teeth,
                            SPUR_J_TEETH, SPUR_J_MATING, SPUR_J,
                            log_axes=(True, True))
    j75 = _interp_grid(teeth, helix_angle,
                       HELICAL_J_TEETH, HELICAL_J_ANGLES, HELICAL_J75,
                       log_axes=(True, False))
    mult = _interp_grid(mating_teeth, helix_angle,
                        HELICAL_JMULT_TEETH, HELICAL_J_ANGLES,
                        HELICAL_J_MULTIPLIER, log_axes=(True, False))
    return j75 * mult


# ----------------------------------------------------------------------
# Load-distribution (Cma) coefficients — Shigley Table 14-9.
# Cma = A + B*F + C*F^2 with the face width F in INCHES.
# ----------------------------------------------------------------------

CMA_COEFFICIENTS = {
    "open": (0.247, 0.0167, -0.765e-4),
    "commercial": (0.127, 0.0158, -0.930e-4),
    "precision": (0.0675, 0.0128, -0.926e-4),
    "extra_precision": (0.00360, 0.0102, -0.822e-4),
}

# ----------------------------------------------------------------------
# Size factor Ks by module — stepped metric table.
# (upper module bound of the band, Ks). A tooth larger than the band's
# bound falls into the next band; the last value covers everything above.
# ----------------------------------------------------------------------

SIZE_FACTOR_BY_MODULE = (
    (5.0, 1.00),
    (6.0, 1.05),
    (8.0, 1.15),
    (12.0, 1.25),
    (20.0, 1.40),
)

# ----------------------------------------------------------------------
# Reliability factor YZ (KR) — Shigley Table 14-10.
# ----------------------------------------------------------------------

RELIABILITY_YZ = {
    0.50: 0.70,
    0.90: 0.85,
    0.99: 1.00,
    0.999: 1.25,
    0.9999: 1.50,
}


def reliability_factor(reliability=0.99):
    """
    Reliability factor YZ (KR).

    Uses Table 14-10 values exactly and the Shigley log fits between
    them: ``0.658 - 0.0759 ln(1 - R)`` for 0.5 <= R < 0.99 and
    ``0.50 - 0.109 ln(1 - R)`` for 0.99 <= R <= 0.9999.

    Args:
        reliability (float): Survival probability R (0.5 to 0.9999).

    Returns:
        float: YZ factor.

    Raises:
        ValueError: If ``reliability`` is outside [0.5, 0.9999].
    """
    r = reliability
    if not 0.5 <= r <= 0.9999:
        raise ValueError("Reliability must be between 0.5 and 0.9999")
    if r in RELIABILITY_YZ:
        return RELIABILITY_YZ[r]
    if r < 0.99:
        return 0.658 - 0.0759 * math.log(1 - r)
    return 0.50 - 0.109 * math.log(1 - r)


# ----------------------------------------------------------------------
# Allowable stress numbers for gear steels (metric MPa fits of
# Shigley Figs. 14-2 / 14-5 and Tables 14-3 / 14-6).
# ----------------------------------------------------------------------

def allowable_bending_stress(brinell_hardness, grade=1):
    """
    AGMA allowable bending stress number St for through-hardened steel.

    Metric fits of Fig. 14-2: grade 1 ``0.533 HB + 88.3`` MPa,
    grade 2 ``0.703 HB + 113`` MPa.

    Args:
        brinell_hardness (float): Core hardness HB (roughly 150-400).
        grade (int): AGMA material quality grade, 1 or 2.

    Returns:
        float: St in MPa.

    Raises:
        ValueError: For an unknown grade or non-positive hardness.
    """
    if brinell_hardness <= 0:
        raise ValueError("Hardness must be strictly positive")
    if grade == 1:
        return 0.533 * brinell_hardness + 88.3
    if grade == 2:
        return 0.703 * brinell_hardness + 113.0
    raise ValueError("Grade must be 1 or 2")


def allowable_contact_stress(brinell_hardness, grade=1):
    """
    AGMA allowable contact stress number Sc for through-hardened steel.

    Metric fits of Fig. 14-5: grade 1 ``2.22 HB + 200`` MPa,
    grade 2 ``2.41 HB + 237`` MPa.

    Args:
        brinell_hardness (float): Surface hardness HB.
        grade (int): AGMA material quality grade, 1 or 2.

    Returns:
        float: Sc in MPa.

    Raises:
        ValueError: For an unknown grade or non-positive hardness.
    """
    if brinell_hardness <= 0:
        raise ValueError("Hardness must be strictly positive")
    if grade == 1:
        return 2.22 * brinell_hardness + 200.0
    if grade == 2:
        return 2.41 * brinell_hardness + 237.0
    raise ValueError("Grade must be 1 or 2")


#: Named case-hardened gear steels: {name: (St MPa, Sc MPa)}.
#: Metric equivalents of Tables 14-3 / 14-6 entries.
GEAR_STEELS = {
    "carburized_grade1": {"St": 380.0, "Sc": 1240.0},
    "carburized_grade2": {"St": 450.0, "Sc": 1550.0},
    "carburized_grade3": {"St": 515.0, "Sc": 1895.0},
    "nitrided_grade1": {"St": 285.0, "Sc": 1050.0},
    "nitrided_grade2": {"St": 325.0, "Sc": 1190.0},
}

# ----------------------------------------------------------------------
# Worm-drive Buckingham load-stress (wear) factors K in N/mm^2 for
# W_allow = K * d_wheel * face_width. Rough educational values.
# ----------------------------------------------------------------------

WORM_WEAR_K = {
    "steel_bronze": 0.415,          # 250 HB steel worm / bronze wheel
    "hardened_steel_bronze": 0.553,  # hardened worm / sand-cast bronze
    "bronze_chilled": 0.830,        # hardened worm / chill-cast bronze
    "cast_iron_bronze": 1.035,      # cast-iron worm / bronze wheel
}
