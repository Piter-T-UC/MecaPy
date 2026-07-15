"""ISO metric thread and bolt property-class data.

Tabulated values for ISO metric coarse threads (ISO 262 / ISO 724) and
bolt property classes (ISO 898-1). Units: mm for dimensions, mm^2 for
areas, MPa for strengths.
"""

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
