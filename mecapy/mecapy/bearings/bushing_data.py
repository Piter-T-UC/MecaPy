"""Boundary-lubricated bushing material data (Shigley Table 12-8).

Provenance: Shigley's *Mechanical Engineering Design*, Table 12-8, "Some
materials for boundary-lubricated bearings", converted from US customary
to SI.  The four rated columns (maximum load, maximum temperature,
maximum speed, maximum PV) are transcribed and marked ``confirmed``;
running friction coefficients are **not** in that table and are marked
``ESTIMATED`` typical values.

Wear factors are deliberately absent: wear depends on the counterface
material, hardness and finish as much as on the bushing, so
:meth:`~mecapy.bearings.plain.PlainBearing.wear_depth` requires the
factor to be supplied from test or supplier data rather than guessed
from a table.

Units: pressure in MPa, velocity in m/s, PV in MPa*m/s, temperature in
degrees Celsius.  Friction coefficients are dimensionless.

Fields per material:
    p_max (float): Maximum bearing pressure in MPa.
    v_max (float): Maximum rubbing velocity in m/s.
    pv_max (float): Maximum PV product in MPa*m/s.
    t_max (float): Maximum operating temperature in degrees Celsius.
    mu (float): Representative running friction coefficient.
"""

#: Boundary-lubricated bushing materials (Shigley Table 12-8, SI).
BUSHING_MATERIALS = {
    "cast_bronze": {
        # 4500 psi, 1500 fpm, 50 000 psi-fpm, 325 degF (confirmed)
        "p_max": 31.0,
        "v_max": 7.62,
        "pv_max": 1.75,
        "t_max": 163.0,
        "mu": 0.10,  # ESTIMATED
    },
    "porous_bronze": {
        # 4500 psi, 1500 fpm, 50 000 psi-fpm, 150 degF (confirmed)
        "p_max": 31.0,
        "v_max": 7.62,
        "pv_max": 1.75,
        "t_max": 66.0,
        "mu": 0.12,  # ESTIMATED
    },
    "porous_iron": {
        # 8000 psi, 800 fpm, 50 000 psi-fpm, 150 degF (confirmed)
        "p_max": 55.2,
        "v_max": 4.06,
        "pv_max": 1.75,
        "t_max": 66.0,
        "mu": 0.12,  # ESTIMATED
    },
    "ptfe": {
        # 500 psi, 50 fpm, 1000 psi-fpm, 500 degF (confirmed)
        "p_max": 3.45,
        "v_max": 0.25,
        "pv_max": 0.035,
        "t_max": 260.0,
        "mu": 0.05,  # ESTIMATED (lowest of any bushing material)
    },
    "filled_ptfe": {
        # 2500 psi, 1000 fpm, 10 000 psi-fpm, 500 degF (confirmed)
        "p_max": 17.2,
        "v_max": 5.08,
        "pv_max": 0.35,
        "t_max": 260.0,
        "mu": 0.10,  # ESTIMATED
    },
    "ptfe_fabric": {
        # 60 000 psi, 50 fpm, 25 000 psi-fpm, 500 degF (confirmed)
        "p_max": 414.0,
        "v_max": 0.25,
        "pv_max": 0.88,
        "t_max": 260.0,
        "mu": 0.08,  # ESTIMATED
    },
    "nylon": {
        # 1000 psi, 1000 fpm, 3000 psi-fpm, 200 degF (confirmed)
        "p_max": 6.89,
        "v_max": 5.08,
        "pv_max": 0.105,
        "t_max": 93.0,
        "mu": 0.20,  # ESTIMATED
    },
    "acetal": {
        # 1000 psi, 1000 fpm, 3000 psi-fpm, 180 degF (confirmed)
        "p_max": 6.89,
        "v_max": 5.08,
        "pv_max": 0.105,
        "t_max": 82.0,
        "mu": 0.20,  # ESTIMATED
    },
    "polycarbonate": {
        # 1000 psi, 1000 fpm, 3000 psi-fpm, 220 degF (confirmed)
        "p_max": 6.89,
        "v_max": 5.08,
        "pv_max": 0.105,
        "t_max": 104.0,
        "mu": 0.25,  # ESTIMATED
    },
    "phenolics": {
        # 6000 psi, 2500 fpm, 15 000 psi-fpm, 200 degF (confirmed)
        "p_max": 41.4,
        "v_max": 12.7,
        "pv_max": 0.525,
        "t_max": 93.0,
        "mu": 0.15,  # ESTIMATED
    },
    "carbon_graphite": {
        # 600 psi, 2500 fpm, 15 000 psi-fpm, 750 degF (confirmed)
        "p_max": 4.14,
        "v_max": 12.7,
        "pv_max": 0.525,
        "t_max": 399.0,
        "mu": 0.20,  # ESTIMATED
    },
    "wood": {
        # 2000 psi, 2000 fpm, 15 000 psi-fpm, 150 degF (confirmed)
        "p_max": 13.8,
        "v_max": 10.16,
        "pv_max": 0.525,
        "t_max": 66.0,
        "mu": 0.25,  # ESTIMATED
    },
}

#: Friction coefficient used when neither an explicit value nor a
#: material is given.  Mid-range for a boundary-lubricated metal bushing.
DEFAULT_BUSHING_MU = 0.12


def get_bushing_material(material):
    """
    Look up design limits for a named bushing material.

    Args:
        material (str): Material name, one of the keys of
            :data:`BUSHING_MATERIALS` (e.g. "cast_bronze", "ptfe").

    Returns:
        dict: ``p_max`` (MPa), ``v_max`` (m/s), ``pv_max`` (MPa*m/s),
        ``t_max`` (degrees C) and ``mu`` (dimensionless) for the material.

    Raises:
        ValueError: If ``material`` is not a known bushing material.
    """
    if material not in BUSHING_MATERIALS:
        available = ", ".join(sorted(BUSHING_MATERIALS))
        raise ValueError(
            f"Unknown bushing material {material!r}. Available: {available}"
        )
    return BUSHING_MATERIALS[material]
