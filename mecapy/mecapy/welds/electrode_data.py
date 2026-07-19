"""Weld electrode strength data and the AISC allowable-stress basis.

Minimum weld-metal properties for the common AWS/ASTM electrode series,
from Shigley's *Mechanical Engineering Design* Table 9-3 (kpsi values
converted to MPa). The AISC allowable shear stress on the weld throat is
0.30 times the electrode tensile strength (Shigley Table 9-4).

Units: strengths in MPa.
"""

# Minimum weld-metal properties (Shigley Table 9-3). Strengths in MPa,
# rounded from the tabulated kpsi values (e.g. E70xx: 70 kpsi -> 482 MPa,
# 57 kpsi -> 393 MPa).
ELECTRODES = {
    "E60xx": {"tensile_strength": 427.0, "yield_strength": 345.0},
    "E70xx": {"tensile_strength": 482.0, "yield_strength": 393.0},
    "E80xx": {"tensile_strength": 551.0, "yield_strength": 462.0},
    "E90xx": {"tensile_strength": 620.0, "yield_strength": 531.0},
    "E100xx": {"tensile_strength": 689.0, "yield_strength": 600.0},
    "E120xx": {"tensile_strength": 827.0, "yield_strength": 745.0},
}

# AISC allowable shear on the weld throat as a fraction of the electrode
# tensile strength (Shigley Table 9-4): tau_allow = 0.30 * S_ut.
AISC_SHEAR_ALLOWABLE_FACTOR = 0.30


def get_electrode(name):
    """
    Look up minimum weld-metal properties for an electrode designation.

    The trailing ``xx`` (arc/position digits) is optional and the lookup
    is case-insensitive, so "E70", "e70" and "E70xx" all resolve to the
    same entry.

    Args:
        name (str): Electrode designation, e.g. "E70" or "E70xx".

    Returns:
        dict: ``tensile_strength`` and ``yield_strength`` in MPa.

    Raises:
        ValueError: If ``name`` is not a known electrode.
    """
    key = str(name).upper()
    if not key.endswith("XX"):
        key += "XX"
    # Normalize to the stored capitalization ("E70XX" -> "E70xx").
    key = key[:-2] + "xx"
    if key not in ELECTRODES:
        available = ", ".join(ELECTRODES.keys())
        raise ValueError(
            f"Unknown electrode {name!r}. Available electrodes: {available}"
        )
    return ELECTRODES[key]
