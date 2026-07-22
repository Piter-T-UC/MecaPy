"""Static failure criteria (Shigley Ch. 5).

Standalone, unit-agnostic helper functions for static strength checks: the
ductile criteria (maximum-shear-stress and distortion-energy / von Mises)
and the brittle criteria (Coulomb-Mohr and modified Mohr). Every function
is pure — it takes stresses and strengths and returns a number, storing
nothing.

Units are the caller's: pass stresses and strengths in the same unit (all
MPa, or all Pa) and the safety factors come out dimensionless. Strengths
(``Sy``, ``Sut``, ``Suc``) are magnitudes and must be strictly positive;
``Suc`` is the magnitude of the ultimate compressive strength (a positive
number). Principal stresses are signed: tension positive, compression
negative, ordered ``sigma_1 >= sigma_3``.
"""

import math


def von_mises(sx=0.0, sy=0.0, sz=0.0, txy=0.0, tyz=0.0, tzx=0.0):
    """
    von Mises (distortion-energy) equivalent stress (Shigley Eq. 5-14).

    ``sigma' = sqrt( ((sx-sy)^2 + (sy-sz)^2 + (sz-sx)^2)/2
                     + 3(txy^2 + tyz^2 + tzx^2) )``

    Args:
        sx, sy, sz (float): Normal stresses on the x/y/z faces.
        txy, tyz, tzx (float): Shear stresses.

    Returns:
        float: Equivalent (von Mises) stress in the input unit.
    """
    return math.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)
    )


def principal_stresses(sx=0.0, sy=0.0, txy=0.0):
    """
    Ordered principal stresses of a plane-stress state.

    The two in-plane principals ``(sx+sy)/2 +/- sqrt(((sx-sy)/2)^2 + txy^2)``
    together with the out-of-plane principal 0, sorted descending.

    Args:
        sx, sy (float): In-plane normal stresses.
        txy (float): In-plane shear stress.

    Returns:
        tuple: ``(sigma_1, sigma_2, sigma_3)`` with
        ``sigma_1 >= sigma_2 >= sigma_3``.
    """
    center = (sx + sy) / 2.0
    radius = math.sqrt(((sx - sy) / 2.0) ** 2 + txy ** 2)
    a, b = center + radius, center - radius
    return tuple(sorted((a, b, 0.0), reverse=True))


def n_distortion_energy(yield_strength, sx=0.0, sy=0.0, sz=0.0,
                        txy=0.0, tyz=0.0, tzx=0.0):
    """
    Ductile safety factor by the distortion-energy (DE) theory.

    ``n = Sy / sigma'`` (Shigley Eq. 5-19), with ``sigma'`` the von Mises
    stress. The standard ductile criterion.

    Args:
        yield_strength (float): Yield strength Sy (> 0).
        sx, sy, sz, txy, tyz, tzx (float): Stress components (see
            :func:`von_mises`).

    Returns:
        float: Safety factor against yielding; ``math.inf`` at zero stress.

    Raises:
        ValueError: If ``yield_strength`` is not strictly positive.
    """
    if yield_strength <= 0:
        raise ValueError("Yield strength must be strictly positive")
    sigma = von_mises(sx, sy, sz, txy, tyz, tzx)
    if sigma == 0:
        return math.inf
    return yield_strength / sigma


def n_maximum_shear_stress(yield_strength, sigma_1, sigma_3):
    """
    Ductile safety factor by the maximum-shear-stress (MSS) theory.

    ``n = Sy / (sigma_1 - sigma_3)`` (Shigley Eq. 5-3), the more
    conservative ductile criterion (Tresca).

    Args:
        yield_strength (float): Yield strength Sy (> 0).
        sigma_1 (float): Largest principal stress.
        sigma_3 (float): Smallest principal stress (``<= sigma_1``).

    Returns:
        float: Safety factor against yielding; ``math.inf`` when
        ``sigma_1 == sigma_3``.

    Raises:
        ValueError: If ``yield_strength`` is not strictly positive or
            ``sigma_1 < sigma_3``.
    """
    if yield_strength <= 0:
        raise ValueError("Yield strength must be strictly positive")
    if sigma_1 < sigma_3:
        raise ValueError("Require sigma_1 >= sigma_3")
    if sigma_1 == sigma_3:
        return math.inf
    return yield_strength / (sigma_1 - sigma_3)


def n_coulomb_mohr(sigma_1, sigma_3, ultimate_tensile, ultimate_compressive):
    """
    Brittle safety factor by the Coulomb-Mohr theory (Shigley Eq. 5-26).

    Three-branch form for a plane-stress state (``sigma_2 = 0`` folded into
    the principal ordering):

    - both tensile  (``sigma_1 >= sigma_3 >= 0``):  ``n = Sut / sigma_1``
    - mixed (``sigma_1 >= 0 >= sigma_3``):
      ``1/n = sigma_1/Sut - sigma_3/Suc``
    - both compressive (``0 >= sigma_1 >= sigma_3``): ``n = Suc / |sigma_3|``

    Args:
        sigma_1 (float): Largest principal stress (signed).
        sigma_3 (float): Smallest principal stress (``<= sigma_1``).
        ultimate_tensile (float): Ultimate tensile strength Sut (> 0).
        ultimate_compressive (float): Magnitude of the ultimate compressive
            strength Suc (> 0).

    Returns:
        float: Safety factor against fracture; ``math.inf`` at zero stress.

    Raises:
        ValueError: If a strength is not strictly positive or
            ``sigma_1 < sigma_3``.
    """
    sut, suc = _check_brittle_strengths(ultimate_tensile, ultimate_compressive)
    if sigma_1 < sigma_3:
        raise ValueError("Require sigma_1 >= sigma_3")
    if sigma_1 <= 0 and sigma_3 == 0:
        return math.inf
    if sigma_1 >= 0 and sigma_3 >= 0:
        return math.inf if sigma_1 == 0 else sut / sigma_1
    if sigma_1 >= 0 and sigma_3 < 0:
        return 1.0 / (sigma_1 / sut - sigma_3 / suc)
    return suc / (-sigma_3)  # both compressive


def n_modified_mohr(sigma_1, sigma_3, ultimate_tensile, ultimate_compressive):
    """
    Brittle safety factor by the modified-Mohr theory (Shigley Eq. 5-32).

    Less conservative than Coulomb-Mohr in the fourth quadrant and the
    better match to gray-cast-iron data. With ``sigma_A = sigma_1``,
    ``sigma_B = sigma_3`` (``sigma_A >= sigma_B``):

    - ``sigma_B >= 0``:           ``n = Sut / sigma_1``
    - ``sigma_A <= 0``:           ``n = Suc / |sigma_3|``
    - ``sigma_A > 0 > sigma_B`` and ``|sigma_B| <= sigma_A``:
      ``n = Sut / sigma_1``
    - ``sigma_A > 0 > sigma_B`` and ``|sigma_B| > sigma_A``:
      ``1/n = (Suc - Sut) sigma_1 / (Suc Sut) - sigma_3 / Suc``

    Args:
        sigma_1 (float): Largest principal stress (signed).
        sigma_3 (float): Smallest principal stress (``<= sigma_1``).
        ultimate_tensile (float): Ultimate tensile strength Sut (> 0).
        ultimate_compressive (float): Magnitude of Suc (> 0).

    Returns:
        float: Safety factor against fracture; ``math.inf`` at zero stress.

    Raises:
        ValueError: If a strength is not strictly positive or
            ``sigma_1 < sigma_3``.
    """
    sut, suc = _check_brittle_strengths(ultimate_tensile, ultimate_compressive)
    if sigma_1 < sigma_3:
        raise ValueError("Require sigma_1 >= sigma_3")
    if sigma_3 >= 0:  # sigma_1 >= sigma_3 >= 0
        return math.inf if sigma_1 == 0 else sut / sigma_1
    if sigma_1 <= 0:  # both compressive
        return suc / (-sigma_3)
    # sigma_1 > 0 > sigma_3
    if -sigma_3 <= sigma_1:
        return sut / sigma_1
    return 1.0 / ((suc - sut) * sigma_1 / (suc * sut) - sigma_3 / suc)


def _check_brittle_strengths(ultimate_tensile, ultimate_compressive):
    """Validate and return the (Sut, Suc) magnitudes for the brittle theories."""
    if ultimate_tensile <= 0:
        raise ValueError("Ultimate tensile strength must be strictly positive")
    if ultimate_compressive <= 0:
        raise ValueError(
            "Ultimate compressive strength (magnitude) must be strictly positive"
        )
    return ultimate_tensile, ultimate_compressive
