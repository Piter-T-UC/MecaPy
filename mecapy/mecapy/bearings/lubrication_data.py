"""Lubricant viscosity and hydrodynamic bearing chart data (Shigley ch. 12).

Viscosity-temperature fits after Seireg & Dandage as given with Shigley
fig. 12-13; journal-bearing chart data from Raimondi & Boyd (Trans. ASLE 1,
1958) as reproduced in Shigley figs. 12-16 to 12-24.

Units: viscosity in mPa*s (= cP), temperature in degrees Celsius unless
noted otherwise.  Chart-derived values carry the usual few-percent
digitization tolerance of figure-read data.
"""

import math

import numpy as np

#: Seireg & Dandage viscosity fit mu = mu0 * exp(b / (T_F + 95)) with
#: mu0 in microreyn and T_F in degrees Fahrenheit (Shigley fig. 12-13).
SAE_VISCOSITY_CONSTANTS = {
    10: (0.0158, 1157.5),
    20: (0.0136, 1271.6),
    30: (0.0141, 1360.0),
    40: (0.0121, 1474.4),
    50: (0.0170, 1509.6),
    60: (0.0187, 1564.0),
}

#: 1 microreyn = 6.89 mPa*s (1 reyn = 6890 Pa*s).
MPAS_PER_MICROREYN = 6.89

#: Bearing modulus mu*N/P at the thick-film boundary (mu in cP = mPa*s,
#: N in rpm, P in psi), the McKee transition of Shigley fig. 12-4.  Below
#: this the film is boundary/mixed ("unstable"): friction rises as the
#: viscosity drops further.
STABILITY_BOUNDARY_BEARING_MODULUS = 30.0

#: Trumpler design criteria (Shigley sec. 12-14, converted to SI):
#: minimum film h0 >= 0.005 mm + 0.00004 * d (d = journal diameter, mm);
#: maximum lubricant temperature 121 degC; starting pressure
#: W_st / (2 r l) <= 2.07 MPa; design factor n_d >= 2 on the load.
TRUMPLER_MIN_FILM_BASE = 0.005  # mm
TRUMPLER_MIN_FILM_SLOPE = 0.00004  # mm per mm of journal diameter
TRUMPLER_MAX_TEMPERATURE = 121.0  # degC
TRUMPLER_MAX_STARTUP_PRESSURE = 2.07  # MPa

#: Raimondi-Boyd design charts, digitized (Shigley figs. 12-16 to 12-24).
#: Keyed by l/d; rows are
#: (h0/c, S, phi [deg], (r/c)f, Q/(r c N l), Qs/Q, P/pmax).
RAIMONDI_BOYD = {
    "inf": [
        (0.9, 0.240, 69.10, 4.80, 3.03, 0.0, 0.826),
        (0.8, 0.123, 67.26, 2.57, 2.83, 0.0, 0.814),
        (0.6, 0.0626, 61.94, 1.52, 2.26, 0.0, 0.764),
        (0.4, 0.0389, 54.31, 1.20, 1.56, 0.0, 0.667),
        (0.2, 0.021, 42.22, 0.961, 0.760, 0.0, 0.495),
        (0.1, 0.0115, 31.62, 0.756, 0.411, 0.0, 0.358),
    ],
    1.0: [
        (0.9, 1.33, 79.5, 26.4, 3.37, 0.150, 0.540),
        (0.8, 0.631, 74.02, 12.8, 3.59, 0.280, 0.529),
        (0.6, 0.264, 63.10, 5.79, 3.99, 0.497, 0.484),
        (0.4, 0.121, 50.58, 3.22, 4.33, 0.680, 0.415),
        (0.2, 0.0446, 36.24, 1.70, 4.62, 0.842, 0.313),
        (0.1, 0.0188, 26.45, 1.05, 4.74, 0.919, 0.247),
        (0.03, 0.00474, 15.47, 0.514, 4.82, 0.973, 0.152),
    ],
    0.5: [
        (0.9, 4.31, 81.62, 85.6, 3.43, 0.173, 0.523),
        (0.8, 2.03, 74.94, 40.9, 3.72, 0.318, 0.506),
        (0.6, 0.779, 61.45, 17.0, 4.29, 0.552, 0.441),
        (0.4, 0.319, 48.14, 8.10, 4.85, 0.730, 0.365),
        (0.2, 0.0923, 33.31, 3.26, 5.41, 0.874, 0.267),
        (0.1, 0.0313, 23.66, 1.60, 5.69, 0.939, 0.206),
        (0.03, 0.00609, 13.75, 0.610, 5.88, 0.980, 0.126),
    ],
    0.25: [
        (0.9, 16.2, 82.31, 322.0, 3.45, 0.180, 0.515),
        (0.8, 7.57, 75.18, 153.0, 3.76, 0.330, 0.489),
        (0.6, 2.83, 60.86, 61.1, 4.37, 0.567, 0.415),
        (0.4, 1.07, 46.72, 26.7, 4.99, 0.746, 0.334),
        (0.2, 0.261, 31.04, 8.80, 5.60, 0.884, 0.240),
        (0.1, 0.0736, 21.85, 3.50, 5.91, 0.945, 0.180),
        (0.03, 0.0101, 12.22, 0.922, 6.12, 0.984, 0.108),
    ],
}

#: Field names of the raimondi_boyd() result, in table-column order.
_RB_FIELDS = [
    "h0_over_c",
    "phi_deg",
    "friction_variable",
    "flow_variable",
    "side_flow_ratio",
    "p_over_pmax",
]


def viscosity(sae_grade, temperature):
    """Absolute viscosity of an SAE-grade oil at temperature (fig. 12-13).

    Uses the Seireg & Dandage closed-form fit
    mu = mu0 * exp(b / (T_F + 95)), converted to SI on the way out.

    Args:
        sae_grade (int): SAE grade, one of 10, 20, 30, 40, 50, 60.
        temperature (float): Oil temperature in degrees Celsius.

    Returns:
        float: Absolute viscosity in mPa*s (= cP).

    Raises:
        ValueError: If the grade is unknown.
    """
    temperature_f = 1.8 * temperature + 32.0
    return MPAS_PER_MICROREYN * viscosity_reyn(sae_grade, temperature_f)


def viscosity_reyn(sae_grade, temperature_f):
    """Absolute viscosity in microreyn at a Fahrenheit temperature.

    Native imperial form of the Seireg & Dandage fit, convenient for
    cross-checking Shigley worked examples.

    Args:
        sae_grade (int): SAE grade, one of 10, 20, 30, 40, 50, 60.
        temperature_f (float): Oil temperature in degrees Fahrenheit.

    Returns:
        float: Absolute viscosity in microreyn.

    Raises:
        ValueError: If the grade is unknown.
    """
    if sae_grade not in SAE_VISCOSITY_CONSTANTS:
        raise ValueError(
            f"Unknown SAE grade {sae_grade}. "
            f"Available: {sorted(SAE_VISCOSITY_CONSTANTS)}"
        )
    mu0, b = SAE_VISCOSITY_CONSTANTS[sae_grade]
    return mu0 * math.exp(b / (temperature_f + 95.0))


def is_thick_film(viscosity_mpas, speed_rps, pressure_mpa):
    """Whether operation lies in the stable thick-film regime (fig. 12-4).

    Compares the McKee bearing modulus mu*N/P (mu in cP, N in rpm, P in
    psi) against the empirical transition value of about 30; below it
    lies boundary or mixed-film ("unstable") lubrication where friction
    rises as viscosity drops.

    Args:
        viscosity_mpas (float): Absolute viscosity in mPa*s (= cP).
        speed_rps (float): Journal speed in rev/s.
        pressure_mpa (float): Projected-area pressure P = W/(2*r*l) in MPa.

    Returns:
        bool: True for hydrodynamic (thick-film) operation.

    Raises:
        ValueError: For non-positive inputs.
    """
    if viscosity_mpas <= 0 or speed_rps <= 0 or pressure_mpa <= 0:
        raise ValueError("Viscosity, speed and pressure must be strictly positive")
    speed_rpm = speed_rps * 60.0
    pressure_psi = pressure_mpa * 145.038
    modulus = viscosity_mpas * speed_rpm / pressure_psi
    return modulus >= STABILITY_BOUNDARY_BEARING_MODULUS


def _interp_table(sommerfeld, table):
    """Interpolate every chart variable vs log10(S) in one l/d table."""
    rows = sorted(table, key=lambda row: row[1])
    log_s = [math.log10(row[1]) for row in rows]
    x = math.log10(sommerfeld)
    result = {}
    for i, field in enumerate(_RB_FIELDS):
        col_index = 0 if i == 0 else i + 1  # skip the S column
        values = [row[col_index] for row in rows]
        result[field] = float(np.interp(x, log_s, values))
    return result


def raimondi_boyd(sommerfeld, l_over_d):
    """Raimondi-Boyd chart variables for a bearing operating point.

    Each variable is interpolated against log10(S) within the digitized
    tables (clamped at the tabulated ends, as the charts themselves are
    bounded).  For l/d ratios between the tabulated 1/4, 1/2, 1 and
    infinity, Shigley eq. 12-16 (the Raimondi-Boyd cubic weighting in
    L = l/d) blends the four tables; it reduces exactly to each table at
    L = 1/4, 1/2 and 1.

    Args:
        sommerfeld (float): Sommerfeld number S = (r/c)**2 * mu*N/P
            (dimensionless, consistent units).
        l_over_d (float): Bearing length-to-diameter ratio l/d, >= 0.25.
            Use ``math.inf`` for the infinite (long) bearing solution.

    Returns:
        dict: ``h0_over_c`` (minimum film thickness ratio), ``phi_deg``
        (position of minimum film), ``friction_variable`` ((r/c)f),
        ``flow_variable`` (Q/(r c N l)), ``side_flow_ratio`` (Qs/Q) and
        ``p_over_pmax`` (film pressure ratio).

    Raises:
        ValueError: If S is non-positive or l/d < 0.25 (below the chart
            range; Shigley suggests such bearings be redesigned).
    """
    if sommerfeld <= 0:
        raise ValueError("Sommerfeld number must be strictly positive")
    if l_over_d != math.inf and l_over_d < 0.25:
        raise ValueError(
            "l/d must be at least 0.25 (below the Raimondi-Boyd chart range)"
        )
    if l_over_d == math.inf:
        return _interp_table(sommerfeld, RAIMONDI_BOYD["inf"])
    for key in (1.0, 0.5, 0.25):
        if math.isclose(l_over_d, key, rel_tol=1e-9):
            return _interp_table(sommerfeld, RAIMONDI_BOYD[key])
    # Shigley eq. 12-16: cubic blending of the four tabulated solutions.
    y_inf = _interp_table(sommerfeld, RAIMONDI_BOYD["inf"])
    y_1 = _interp_table(sommerfeld, RAIMONDI_BOYD[1.0])
    y_half = _interp_table(sommerfeld, RAIMONDI_BOYD[0.5])
    y_quarter = _interp_table(sommerfeld, RAIMONDI_BOYD[0.25])
    L = l_over_d
    result = {}
    for field in _RB_FIELDS:
        result[field] = (1.0 / L**3) * (
            -(1.0 / 8.0) * (1 - L) * (1 - 2 * L) * (1 - 4 * L) * y_inf[field]
            + (1.0 / 3.0) * (1 - 2 * L) * (1 - 4 * L) * y_1[field]
            - (1.0 / 4.0) * (1 - L) * (1 - 4 * L) * y_half[field]
            + (1.0 / 24.0) * (1 - L) * (1 - 2 * L) * y_quarter[field]
        )
    return result
