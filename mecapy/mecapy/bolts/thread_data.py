"""ISO metric thread and bolt property-class data.

Tabulated values for ISO metric coarse threads (ISO 262 / ISO 724) and
bolt property classes (ISO 898-1), plus Shigley's closed-form thread
geometry for threads not in the table (fine series, custom pitches).
Units: mm for dimensions, mm^2 for areas, MPa for strengths.
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


def get_thread(size):
    """
    Look up ISO coarse thread data for a thread designation.

    Args:
        size (str): Thread designation, e.g. "M10".

    Returns:
        dict: ``nominal_diameter`` (mm), ``pitch`` (mm) and
        ``stress_area`` (mm^2) for the given size.

    Raises:
        ValueError: If ``size`` is not a known thread designation.
    """
    if size not in ISO_COARSE_THREADS:
        available = ", ".join(ISO_COARSE_THREADS.keys())
        raise ValueError(f"Unknown thread size {size!r}. Available sizes: {available}")
    return ISO_COARSE_THREADS[size]


def get_pitch(size):
    """
    Look up the ISO coarse thread pitch for a size or nominal diameter.

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


def get_property_class(property_class):
    """
    Look up strength data for a bolt property class.

    Args:
        property_class (str): Property class designation, e.g. "8.8".

    Returns:
        dict: ``tensile_strength``, ``yield_strength`` and
        ``proof_strength`` in MPa.

    Raises:
        ValueError: If ``property_class`` is not a known class.
    """
    if property_class not in PROPERTY_CLASSES:
        available = ", ".join(PROPERTY_CLASSES.keys())
        raise ValueError(
            f"Unknown property class {property_class!r}. Available classes: {available}"
        )
    return PROPERTY_CLASSES[property_class]
