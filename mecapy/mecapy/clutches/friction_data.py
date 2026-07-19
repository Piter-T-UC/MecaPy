"""Friction-lining material data for clutches and brakes.

Representative design values for common friction linings, digitized
from the Shigley Ch. 16 friction-materials table (Table 16-3 region).
Each entry stores a mid-range design value; real linings span a band
around these numbers, so treat them as starting points and confirm
against a supplier datasheet for a final design.

Fields per lining:
    ``mu_dry``  — dry friction coefficient (dimensionless)
    ``mu_oil``  — friction coefficient running in oil, or ``None`` when
                  the lining is not used wet
    ``p_max``   — maximum allowable contact pressure in MPa
    ``t_max``   — maximum continuous operating temperature in degrees C
"""

FRICTION_MATERIALS = {
    "molded": {"mu_dry": 0.35, "mu_oil": 0.06, "p_max": 1.55, "t_max": 260},
    "woven": {"mu_dry": 0.35, "mu_oil": 0.08, "p_max": 0.52, "t_max": 200},
    "sintered_metal": {"mu_dry": 0.30, "mu_oil": 0.06, "p_max": 2.07, "t_max": 450},
    "rigid_molded_nonasbestos": {"mu_dry": 0.38, "mu_oil": None, "p_max": 1.03, "t_max": 260},
    "cermet": {"mu_dry": 0.32, "mu_oil": None, "p_max": 1.03, "t_max": 815},
    "cork": {"mu_dry": 0.35, "mu_oil": 0.18, "p_max": 0.10, "t_max": 82},
    "wood": {"mu_dry": 0.25, "mu_oil": 0.12, "p_max": 0.41, "t_max": 90},
    "cast_iron": {"mu_dry": 0.20, "mu_oil": 0.05, "p_max": 1.72, "t_max": 260},
}


def get_friction_material(lining):
    """
    Look up design values for a named friction lining.

    Args:
        lining (str): Lining name, one of the keys of
            :data:`FRICTION_MATERIALS` (e.g. "molded", "sintered_metal").

    Returns:
        dict: ``mu_dry``, ``mu_oil`` (or None), ``p_max`` (MPa) and
        ``t_max`` (degrees C) for the lining.

    Raises:
        ValueError: If ``lining`` is not a known lining name.
    """
    if lining not in FRICTION_MATERIALS:
        available = ", ".join(sorted(FRICTION_MATERIALS))
        raise ValueError(f"Unknown friction lining {lining!r}. Available: {available}")
    return FRICTION_MATERIALS[lining]
