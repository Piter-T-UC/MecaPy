"""Standalone beam calculation helper functions.

These are convenience functions for quick, closed-form calculations. For a
full model with supports and loads, use :class:`mecapy.beams.Beam`, which is
backed by SymPy.
"""


def calculate_deflection(force, length, elasticity_modulus, moment_of_inertia):
    """
    Maximum deflection of a cantilever beam with an end point load.

    Uses the closed-form relation ``delta = P * L**3 / (3 * E * I)``.

    Args:
        force (float): Applied end force in Newtons.
        length (float): Beam length in meters.
        elasticity_modulus (float): Young's modulus in Pa.
        moment_of_inertia (float): Second moment of area in m^4.

    Returns:
        float: Maximum deflection in meters.
    """
    return force * length ** 3 / (3 * elasticity_modulus * moment_of_inertia)


def calculate_bending_stress(moment, moment_of_inertia, distance):
    """
    Bending stress at a fiber a given distance from the neutral axis.

    Uses ``sigma = M * c / I``.

    Args:
        moment (float): Bending moment in N*m.
        moment_of_inertia (float): Second moment of area in m^4.
        distance (float): Distance ``c`` from the neutral axis in meters.

    Returns:
        float: Bending stress in Pa.
    """
    return moment * distance / moment_of_inertia


def calculate_shear_stress(force, area):
    """
    Average transverse shear stress on a section.

    Uses ``tau = V / A``.

    Args:
        force (float): Shear force in Newtons.
        area (float): Cross-sectional area in m^2.

    Returns:
        float: Shear stress in Pa.
    """
    return force / area
