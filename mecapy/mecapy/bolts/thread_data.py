"""Thread and bolt property-class data (metric and imperial).

Tabulated values for ISO metric coarse threads (ISO 262 / ISO 724) and
bolt property classes (ISO 898-1), for Unified inch threads (ASME B1.1,
UNC and UNF series) and SAE J429 grades, plus Shigley's closed-form
thread geometry for threads not in either table (fine series, custom
pitches).

Units: mm for dimensions, mm^2 for areas, MPa for strengths -- imperial
data is converted once, here in the table, so every consumer downstream
stays metric. The original US customary value is kept in a comment next
to each converted one.
"""

from math import pi

# Shigley's approximations for ISO 60-degree threads (Shigley, Table 8-1):
#   minor diameter  d_r = d - 1.226869 * p
#   pitch diameter  d_p = d - 0.649519 * p
SHIGLEY_MINOR_COEFF = 1.226869
SHIGLEY_PITCH_COEFF = 0.649519

# ISO metric coarse threads: nominal diameter (mm), pitch (mm) and
# tensile stress area As (mm^2) per ISO 898-1 tabulated values.
ISO_COARSE_THREADS = {
    "M3": {"nominal_diameter": 3.0, "pitch": 0.5, "stress_area": 5.03},
    "M4": {"nominal_diameter": 4.0, "pitch": 0.7, "stress_area": 8.78},
    "M5": {"nominal_diameter": 5.0, "pitch": 0.8, "stress_area": 14.2},
    "M6": {"nominal_diameter": 6.0, "pitch": 1.0, "stress_area": 20.1},
    "M8": {"nominal_diameter": 8.0, "pitch": 1.25, "stress_area": 36.6},
    "M10": {"nominal_diameter": 10.0, "pitch": 1.5, "stress_area": 58.0},
    "M12": {"nominal_diameter": 12.0, "pitch": 1.75, "stress_area": 84.3},
    "M14": {"nominal_diameter": 14.0, "pitch": 2.0, "stress_area": 115.0},
    "M16": {"nominal_diameter": 16.0, "pitch": 2.0, "stress_area": 157.0},
    "M18": {"nominal_diameter": 18.0, "pitch": 2.5, "stress_area": 192.0},
    "M20": {"nominal_diameter": 20.0, "pitch": 2.5, "stress_area": 245.0},
    "M22": {"nominal_diameter": 22.0, "pitch": 2.5, "stress_area": 303.0},
    "M24": {"nominal_diameter": 24.0, "pitch": 3.0, "stress_area": 353.0},
    "M27": {"nominal_diameter": 27.0, "pitch": 3.0, "stress_area": 459.0},
    "M30": {"nominal_diameter": 30.0, "pitch": 3.5, "stress_area": 561.0},
    "M33": {"nominal_diameter": 33.0, "pitch": 3.5, "stress_area": 694.0},
    "M36": {"nominal_diameter": 36.0, "pitch": 4.0, "stress_area": 817.0},
}

# Unified inch threads (ASME B1.1), coarse (UNC) and fine (UNF) series,
# from Shigley Table 8-2. Designation is "<size>-<threads per inch>", e.g.
# "1/2-13". Stored in mm and mm^2: d = in * 25.4, p = 25.4 / tpi and
# At = in^2 * 645.16; the tabulated inch value is in the trailing comment.
# Only the fractional sizes 1/4 to 1-1/2 are tabulated -- the range SAE
# J429 grades cover. Numbered machine-screw sizes (#0 to #12) and the
# constant-pitch 8-UN / 12-UN series are reachable through the explicit
# diameter+pitch route of :class:`~mecapy.bolts.bolt.Bolt`.
UNIFIED_THREADS = {
    "1/4-20": {"nominal_diameter": 6.350, "pitch": 1.2700,
               "stress_area": 20.52, "series": "UNC"},  # 0.0318 in^2
    "1/4-28": {"nominal_diameter": 6.350, "pitch": 0.9071,
               "stress_area": 23.48, "series": "UNF"},  # 0.0364 in^2
    "5/16-18": {"nominal_diameter": 7.9375, "pitch": 1.4111,
                "stress_area": 33.81, "series": "UNC"},  # 0.0524 in^2
    "5/16-24": {"nominal_diameter": 7.9375, "pitch": 1.0583,
                "stress_area": 37.42, "series": "UNF"},  # 0.0580 in^2
    "3/8-16": {"nominal_diameter": 9.525, "pitch": 1.5875,
               "stress_area": 50.00, "series": "UNC"},  # 0.0775 in^2
    "3/8-24": {"nominal_diameter": 9.525, "pitch": 1.0583,
               "stress_area": 56.65, "series": "UNF"},  # 0.0878 in^2
    "7/16-14": {"nominal_diameter": 11.1125, "pitch": 1.8143,
                "stress_area": 68.58, "series": "UNC"},  # 0.1063 in^2
    "7/16-20": {"nominal_diameter": 11.1125, "pitch": 1.2700,
                "stress_area": 76.58, "series": "UNF"},  # 0.1187 in^2
    "1/2-13": {"nominal_diameter": 12.700, "pitch": 1.9538,
               "stress_area": 91.55, "series": "UNC"},  # 0.1419 in^2
    "1/2-20": {"nominal_diameter": 12.700, "pitch": 1.2700,
               "stress_area": 103.16, "series": "UNF"},  # 0.1599 in^2
    "9/16-12": {"nominal_diameter": 14.2875, "pitch": 2.1167,
                "stress_area": 117.42, "series": "UNC"},  # 0.182 in^2
    "9/16-18": {"nominal_diameter": 14.2875, "pitch": 1.4111,
                "stress_area": 130.97, "series": "UNF"},  # 0.203 in^2
    "5/8-11": {"nominal_diameter": 15.875, "pitch": 2.3091,
               "stress_area": 145.81, "series": "UNC"},  # 0.226 in^2
    "5/8-18": {"nominal_diameter": 15.875, "pitch": 1.4111,
               "stress_area": 165.16, "series": "UNF"},  # 0.256 in^2
    "3/4-10": {"nominal_diameter": 19.050, "pitch": 2.5400,
               "stress_area": 215.48, "series": "UNC"},  # 0.334 in^2
    "3/4-16": {"nominal_diameter": 19.050, "pitch": 1.5875,
               "stress_area": 240.64, "series": "UNF"},  # 0.373 in^2
    "7/8-9": {"nominal_diameter": 22.225, "pitch": 2.8222,
              "stress_area": 298.06, "series": "UNC"},  # 0.462 in^2
    "7/8-14": {"nominal_diameter": 22.225, "pitch": 1.8143,
               "stress_area": 328.39, "series": "UNF"},  # 0.509 in^2
    "1-8": {"nominal_diameter": 25.400, "pitch": 3.1750,
            "stress_area": 390.97, "series": "UNC"},  # 0.606 in^2
    "1-12": {"nominal_diameter": 25.400, "pitch": 2.1167,
             "stress_area": 427.74, "series": "UNF"},  # 0.663 in^2
    "1-1/8-7": {"nominal_diameter": 28.575, "pitch": 3.6286,
                "stress_area": 492.26, "series": "UNC"},  # 0.763 in^2
    "1-1/8-12": {"nominal_diameter": 28.575, "pitch": 2.1167,
                 "stress_area": 552.26, "series": "UNF"},  # 0.856 in^2
    "1-1/4-7": {"nominal_diameter": 31.750, "pitch": 3.6286,
                "stress_area": 625.16, "series": "UNC"},  # 0.969 in^2
    "1-1/4-12": {"nominal_diameter": 31.750, "pitch": 2.1167,
                 "stress_area": 692.26, "series": "UNF"},  # 1.073 in^2
    "1-3/8-6": {"nominal_diameter": 34.925, "pitch": 4.2333,
                "stress_area": 745.16, "series": "UNC"},  # 1.155 in^2
    "1-3/8-12": {"nominal_diameter": 34.925, "pitch": 2.1167,
                 "stress_area": 848.39, "series": "UNF"},  # 1.315 in^2
    "1-1/2-6": {"nominal_diameter": 38.100, "pitch": 4.2333,
                "stress_area": 906.45, "series": "UNC"},  # 1.405 in^2
    "1-1/2-12": {"nominal_diameter": 38.100, "pitch": 2.1167,
                 "stress_area": 1020.00, "series": "UNF"},  # 1.581 in^2
}

# Bolt property classes per ISO 898-1. Strengths in MPa.
PROPERTY_CLASSES = {
    "4.6": {"tensile_strength": 400.0, "yield_strength": 240.0, "proof_strength": 225.0},
    "4.8": {"tensile_strength": 420.0, "yield_strength": 340.0, "proof_strength": 310.0},
    "5.6": {"tensile_strength": 500.0, "yield_strength": 300.0, "proof_strength": 280.0},
    "5.8": {"tensile_strength": 520.0, "yield_strength": 420.0, "proof_strength": 380.0},
    "8.8": {"tensile_strength": 800.0, "yield_strength": 640.0, "proof_strength": 580.0},
    "10.9": {"tensile_strength": 1000.0, "yield_strength": 900.0, "proof_strength": 830.0},
    "12.9": {"tensile_strength": 1200.0, "yield_strength": 1080.0, "proof_strength": 970.0},
}

# SAE J429 grades for Unified inch bolts (Shigley Table 8-9). Strengths in
# MPa (ksi * 6.894757; the tabulated ksi values are in the comments).
# Grades 2 and 5 are size dependent, so each grade is a list of
# ``(max nominal diameter in mm, strengths)`` rows in ascending diameter
# order; :func:`get_property_class` picks the first row that covers the
# bolt's diameter. Grade 4 (studs only) is not tabulated.
SAE_GRADES = {
    "1": [  # 1/4 to 1 1/2 in: Sp 33, Sut 60, Sy 36 ksi
        (38.1, {"tensile_strength": 413.7, "yield_strength": 248.2, "proof_strength": 227.5}),
    ],
    "2": [  # up to 3/4 in: 55/74/57 ksi -- above 3/4 in: 33/60/36 ksi
        (19.05, {"tensile_strength": 510.2, "yield_strength": 393.0, "proof_strength": 379.2}),
        (38.1, {"tensile_strength": 413.7, "yield_strength": 248.2, "proof_strength": 227.5}),
    ],
    "5": [  # up to 1 in: 85/120/92 ksi -- above 1 in: 74/105/81 ksi
        (25.4, {"tensile_strength": 827.4, "yield_strength": 634.3, "proof_strength": 586.1}),
        (38.1, {"tensile_strength": 723.9, "yield_strength": 558.5, "proof_strength": 510.2}),
    ],
    "5.2": [  # 1/4 to 1 in: Sp 85, Sut 120, Sy 92 ksi
        (25.4, {"tensile_strength": 827.4, "yield_strength": 634.3, "proof_strength": 586.1}),
    ],
    "7": [  # 1/4 to 1 1/2 in: Sp 105, Sut 133, Sy 115 ksi
        (38.1, {"tensile_strength": 916.9, "yield_strength": 792.9, "proof_strength": 723.9}),
    ],
    "8": [  # 1/4 to 1 1/2 in: Sp 120, Sut 150, Sy 130 ksi
        (38.1, {"tensile_strength": 1034.2, "yield_strength": 896.3, "proof_strength": 827.4}),
    ],
    "8.2": [  # 1/4 to 1 in: Sp 120, Sut 150, Sy 130 ksi
        (25.4, {"tensile_strength": 1034.2, "yield_strength": 896.3, "proof_strength": 827.4}),
    ],
}

# Shigley Table 8-7: threaded length LT of a standard hex-head cap screw,
# as ``(max bolt length, addend)`` rows in ascending length order. The
# threaded length is LT = 2*d + addend, so the first row whose max length
# covers the bolt gives the rule. Metric rows are mm; the Unified rows are
# the same table's inch column converted (1/4 in and 1/2 in).
METRIC_THREADED_LENGTH_RULE = ((125.0, 6.0), (200.0, 12.0), (float("inf"), 25.0))
UNIFIED_THREADED_LENGTH_RULE = ((152.4, 6.35), (float("inf"), 12.7))

# Shigley Table 8-7 tabulates metric threaded lengths up to this diameter.
MAX_METRIC_THREADED_LENGTH_DIAMETER = 48.0


def threaded_length(diameter, length, series="ISO metric"):
    """
    Threaded length LT of a standard hex-head cap screw (Shigley Table 8-7).

    Metric: LT = 2*d + 6 mm for L <= 125 mm, 2*d + 12 mm for
    125 < L <= 200 mm, and 2*d + 25 mm beyond that. Unified: LT = 2*d +
    1/4 in for L <= 6 in, else 2*d + 1/2 in.

    The result is not capped at ``length``: a bolt shorter than its own
    LT is fully threaded, which callers detect as ``LT >= length``. See
    :meth:`mecapy.bolts.Bolt.segmented_stiffness`.

    Args:
        diameter (float): Nominal bolt diameter d in mm.
        length (float): Bolt length L in mm.
        series (str): Thread series, as reported by
            :attr:`mecapy.bolts.Bolt.thread_series`. "UNC" and "UNF"
            select the Unified rule; anything else uses the metric rule.

    Returns:
        float: Threaded length LT in mm.

    Raises:
        ValueError: If ``diameter`` or ``length`` is not strictly
            positive, or if a metric diameter exceeds the 48 mm covered
            by Table 8-7.
    """
    if diameter <= 0:
        raise ValueError("Nominal diameter must be strictly positive")
    if length <= 0:
        raise ValueError("Bolt length must be strictly positive")
    if series in ("UNC", "UNF"):
        rule = UNIFIED_THREADED_LENGTH_RULE
    else:
        if diameter > MAX_METRIC_THREADED_LENGTH_DIAMETER:
            raise ValueError(
                f"Shigley Table 8-7 covers nominal diameters up to "
                f"{MAX_METRIC_THREADED_LENGTH_DIAMETER} mm; got {diameter} mm"
            )
        rule = METRIC_THREADED_LENGTH_RULE
    for max_length, addend in rule:
        if length <= max_length:
            return 2.0 * diameter + addend
    raise AssertionError("unreachable: the last rule row covers every length")


def normalize_thread_size(size):
    """
    Canonicalize a thread designation to its table key.

    Strips surrounding whitespace and an optional trailing Unified series
    suffix, so "1/2-13 UNC" and "1/2-13" both canonicalize to "1/2-13".

    Args:
        size (str): Thread designation, e.g. "M10" or "1/2-13 UNC".

    Returns:
        str: The key under which the thread is tabulated.

    Raises:
        ValueError: If ``size`` is not a known thread designation.
    """
    if size in ISO_COARSE_THREADS:
        return size
    key = str(size).strip()
    if key.upper().endswith(("UNC", "UNF")):
        key = key[:-3].strip()
    if key in UNIFIED_THREADS:
        return key
    available = ", ".join(list(ISO_COARSE_THREADS) + list(UNIFIED_THREADS))
    raise ValueError(f"Unknown thread size {size!r}. Available sizes: {available}")


def get_thread(size):
    """
    Look up tabulated thread data for a thread designation.

    Searches the ISO metric coarse table first, then the Unified inch
    table. A trailing series suffix is optional, so "1/2-13 UNC" and
    "1/2-13" resolve to the same thread.

    Args:
        size (str): Thread designation, e.g. "M10" or "1/2-13".

    Returns:
        dict: ``nominal_diameter`` (mm), ``pitch`` (mm) and
        ``stress_area`` (mm^2) for the given size; Unified threads also
        carry ``series`` ("UNC" or "UNF").

    Raises:
        ValueError: If ``size`` is not a known thread designation.
    """
    key = normalize_thread_size(size)
    if key in ISO_COARSE_THREADS:
        return ISO_COARSE_THREADS[key]
    return UNIFIED_THREADS[key]


def get_pitch(size):
    """
    Look up the ISO coarse thread pitch for a size or nominal diameter.

    Metric only: the numeric-diameter lookup scans the ISO coarse table,
    where nominal diameters are unambiguous. Use :func:`get_thread` with a
    full designation (e.g. "1/2-13") for Unified inch threads.

    Args:
        size (str or float): Thread designation (e.g. "M10") or nominal
            diameter in mm (e.g. 10).

    Returns:
        float: Thread pitch in mm (ISO coarse series).

    Raises:
        ValueError: If no ISO coarse thread matches ``size``.
    """
    if isinstance(size, str):
        return get_thread(size)["pitch"]
    diameter = float(size)
    for data in ISO_COARSE_THREADS.values():
        if data["nominal_diameter"] == diameter:
            return data["pitch"]
    available = ", ".join(ISO_COARSE_THREADS.keys())
    raise ValueError(
        f"No ISO coarse thread with nominal diameter {size!r} mm. "
        f"Available sizes: {available}"
    )


def shigley_thread_geometry(diameter, pitch):
    """
    Compute thread geometry with Shigley's formulas.

    Uses the approximations for ISO 60-degree threads:
    d_r = d - 1.226869*p, d_p = d - 0.649519*p and the tensile stress
    area At = (pi/4) * ((d_p + d_r)/2)^2 on the mean of the pitch and
    minor diameters. Intended for threads not in the coarse table
    (fine series, custom pitches).

    Args:
        diameter (float): Nominal (major) thread diameter in mm.
        pitch (float): Thread pitch in mm.

    Returns:
        dict: ``nominal_diameter``, ``pitch``, ``minor_diameter``,
        ``pitch_diameter`` (mm) and ``stress_area`` (mm^2).

    Raises:
        ValueError: If ``diameter`` or ``pitch`` is not strictly
            positive, or the pitch is so large that the minor diameter
            would be non-positive.
    """
    if diameter <= 0:
        raise ValueError("Thread diameter must be strictly positive")
    if pitch <= 0:
        raise ValueError("Thread pitch must be strictly positive")
    minor_diameter = diameter - SHIGLEY_MINOR_COEFF * pitch
    pitch_diameter = diameter - SHIGLEY_PITCH_COEFF * pitch
    if minor_diameter <= 0:
        raise ValueError(
            f"Pitch {pitch} mm is too large for diameter {diameter} mm: "
            "minor diameter would be non-positive"
        )
    stress_area = pi / 4 * ((pitch_diameter + minor_diameter) / 2) ** 2
    return {
        "nominal_diameter": float(diameter),
        "pitch": float(pitch),
        "minor_diameter": minor_diameter,
        "pitch_diameter": pitch_diameter,
        "stress_area": stress_area,
    }


def get_property_class(property_class, diameter=None):
    """
    Look up strength data for a bolt property class or SAE grade.

    ISO 898-1 property classes are size independent, so ``diameter`` is
    ignored for them. SAE J429 grades 2 and 5 are derated above a
    threshold diameter; for those, the row covering ``diameter`` is
    returned. When ``diameter`` is omitted the weakest (largest-size) row
    is returned, so a caller that does not know the size can never
    overstate the strength.

    Args:
        property_class (str): ISO 898-1 property class (e.g. "8.8") or
            SAE J429 grade (e.g. "5").
        diameter (float): Nominal bolt diameter in mm, used to pick the
            right row of a size-dependent SAE grade. Optional.

    Returns:
        dict: ``tensile_strength``, ``yield_strength`` and
        ``proof_strength`` in MPa.

    Raises:
        ValueError: If ``property_class`` is not a known class or grade,
            or if ``diameter`` exceeds the largest size the grade covers.
    """
    if property_class in PROPERTY_CLASSES:
        return PROPERTY_CLASSES[property_class]
    if property_class in SAE_GRADES:
        rows = SAE_GRADES[property_class]
        if diameter is None:
            return rows[-1][1]
        for max_diameter, strengths in rows:
            if diameter <= max_diameter:
                return strengths
        raise ValueError(
            f"SAE grade {property_class!r} is not defined for a nominal diameter of "
            f"{diameter} mm (largest tabulated size: {rows[-1][0]} mm)"
        )
    available = ", ".join(list(PROPERTY_CLASSES) + list(SAE_GRADES))
    raise ValueError(
        f"Unknown property class {property_class!r}. Available classes: {available}"
    )
