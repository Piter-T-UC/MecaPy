"""AGMA gear-rating helper functions (metric / SI form).

These implement the standard AGMA 2001/2101 stress-analysis factors as
presented in Shigley's *Mechanical Engineering Design*. Units are SI:
forces in N, lengths in mm, velocities in m/s and stresses in MPa.
"""

import math


def dynamic_factor(quality_number, velocity):
    """
    AGMA dynamic factor Kv.

    Args:
        quality_number (float): AGMA transmission accuracy grade Qv
            (typically 6-12).
        velocity (float): Pitch-line velocity in m/s.

    Returns:
        float: Dynamic factor Kv (>= 1).
    """
    B = 0.25 * (12 - quality_number) ** (2 / 3)
    A = 50 + 56 * (1 - B)
    return ((A + math.sqrt(200 * velocity)) / A) ** B


def elastic_coefficient(elastic_modulus_1, poisson_1,
                        elastic_modulus_2, poisson_2):
    """
    AGMA elastic coefficient ZE (a.k.a. Cp).

    Args:
        elastic_modulus_1 (float): Young's modulus of the pinion in Pa.
        poisson_1 (float): Poisson's ratio of the pinion.
        elastic_modulus_2 (float): Young's modulus of the gear in Pa.
        poisson_2 (float): Poisson's ratio of the gear.

    Returns:
        float: Elastic coefficient in sqrt(MPa).
    """
    # Convert Pa -> MPa so the coefficient comes out in sqrt(MPa).
    e1 = elastic_modulus_1 / 1e6
    e2 = elastic_modulus_2 / 1e6
    denom = math.pi * ((1 - poisson_1 ** 2) / e1 + (1 - poisson_2 ** 2) / e2)
    return math.sqrt(1 / denom)


def pitting_geometry_factor(pressure_angle, gear_ratio, external=True):
    """
    AGMA surface-strength (pitting) geometry factor ZI (a.k.a. I) for spur
    gears.

    Args:
        pressure_angle (float): Transverse pressure angle in degrees.
        gear_ratio (float): Gear ratio mG = NG / NP (>= 1).
        external (bool): ``True`` for external gear meshes, ``False`` for
            internal meshes.

    Returns:
        float: Pitting geometry factor ZI.
    """
    phi = math.radians(pressure_angle)
    # Load-sharing ratio mN = 1 for spur gears.
    ratio_term = gear_ratio / (gear_ratio + 1) if external else gear_ratio / (gear_ratio - 1)
    return (math.cos(phi) * math.sin(phi) / 2) * ratio_term
