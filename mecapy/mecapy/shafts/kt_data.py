"""Stress-concentration factor (Kt) estimate for round shafts.

Units convention: D, d, r in mm; Kt is dimensionless.

This is an EDUCATIONAL APPROXIMATION, not a digitized chart: it reproduces
the qualitative shape of Peterson's published Kt curves (Kt -> 1 as the
fillet/groove radius grows or D -> d; Kt grows for sharper, deeper notches;
torsion runs lower than bending; a groove runs a bit higher than an
equivalent shoulder fillet) but the coefficients have not been checked
against the real chart. Good enough for design exploration; verify against
Peterson's "Stress Concentration Factors" or Shigley's Table A-15 before
relying on it for a production design (same caveat as this codebase's
BevelGear/Worm simplified rating methods).

# ponytail: constructed power-law approximation, not a table digitization
# (ceiling: coefficients aren't chart-verified). Upgrade path: replace
# _kt_estimate with an interpolated lookup into digitized Table A-15 data,
# keeping the get_kt_shoulder_fillet/get_kt_groove signatures unchanged.
"""

import math

_LOADING_FACTORS = {"bending": 1.0, "torsion": 0.75, "axial": 1.05}
_MAX_KT = 3.5


def _kt_estimate(D, d, r, loading, notch_factor):
    if loading not in _LOADING_FACTORS:
        raise ValueError(f"Unknown loading {loading!r}; expected one of {sorted(_LOADING_FACTORS)}")
    if D <= d:
        raise ValueError("D must be strictly greater than d")
    if d <= 0:
        raise ValueError("d must be strictly positive")
    if r <= 0:
        raise ValueError("Fillet/groove radius must be strictly positive")
    k = _LOADING_FACTORS[loading] * notch_factor
    kt = 1.0 + k * math.sqrt(D / d - 1.0) * (d / (2 * r)) ** 0.35
    return min(kt, _MAX_KT)


def get_kt_shoulder_fillet(D, d, r, loading="bending"):
    """
    Approximate stress-concentration factor for a round-shaft shoulder
    (diameter step) with a fillet.

    Args:
        D (float): Larger (shoulder) diameter, mm.
        d (float): Smaller diameter, mm. The stress is referenced to this
            net section.
        r (float): Fillet radius, mm.
        loading (str): "bending", "torsion", or "axial".

    Returns:
        float: Kt, dimensionless, >= 1.

    Raises:
        ValueError: If D <= d, d <= 0, r <= 0, or loading is unrecognized.
    """
    return _kt_estimate(D, d, r, loading, notch_factor=1.0)


def get_kt_groove(D, d, r, loading="bending"):
    """
    Approximate stress-concentration factor for a round-bottom
    (semicircular) groove in a round shaft.

    Args:
        D (float): Shaft diameter away from the groove, mm.
        d (float): Groove root diameter, mm. The stress is referenced to
            this net section.
        r (float): Groove corner radius, mm.
        loading (str): "bending", "torsion", or "axial".

    Returns:
        float: Kt, dimensionless, >= 1.

    Raises:
        ValueError: If D <= d, d <= 0, r <= 0, or loading is unrecognized.
    """
    return _kt_estimate(D, d, r, loading, notch_factor=1.15)
