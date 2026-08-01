"""Bearing design and analysis module.

Four families, each with its own governing model:

* :class:`Bearing` -- rolling contact, Shigley ch. 11 rating life plus
  ISO 281 modified rating life and ISO 76 static rating.
* :class:`JournalBearing` -- hydrodynamic plain journal, Shigley ch. 12
  (Petroff, Sommerfeld, Raimondi-Boyd, Trumpler).
* :class:`PlainBearing` -- boundary-lubricated bushing rated on PV.
* :class:`ThrustBearing` -- hydrodynamic fixed-incline thrust pad.
"""

from .bearing import Bearing
from .bearing_data import (
    APPLICATION_FACTORS,
    LIFE_EXPONENTS,
    XY_TABLE_TYPES,
    get_application_factor,
    get_life_exponent,
    get_xy_factors,
    weibull_life_multiplier,
    weibull_reliability,
)
from .bushing_data import BUSHING_MATERIALS, get_bushing_material
from .iso281_data import (
    CONTAMINATION_LEVELS,
    LIMITING_SPEED_FACTORS,
    RELIABILITY_FACTORS_A1,
    STATIC_LOAD_FACTORS,
    a_iso,
    get_contamination_factor,
    get_limiting_dn,
    get_reliability_factor,
    get_static_factors,
    reference_viscosity,
    viscosity_ratio,
)
from .journal import JournalBearing
from .lubrication_data import (
    RAIMONDI_BOYD,
    SAE_VISCOSITY_CONSTANTS,
    is_thick_film,
    raimondi_boyd,
    sommerfeld_for,
    viscosity,
    viscosity_reyn,
)
from .plain import PlainBearing
from .thrust import ThrustBearing, load_coefficient

__all__ = [
    # Elements
    "Bearing",
    "JournalBearing",
    "PlainBearing",
    "ThrustBearing",
    # Rolling-contact data (Shigley ch. 11)
    "APPLICATION_FACTORS",
    "LIFE_EXPONENTS",
    "XY_TABLE_TYPES",
    "get_application_factor",
    "get_life_exponent",
    "get_xy_factors",
    "weibull_life_multiplier",
    "weibull_reliability",
    # Rolling-contact data (ISO 281 / ISO 76)
    "CONTAMINATION_LEVELS",
    "LIMITING_SPEED_FACTORS",
    "RELIABILITY_FACTORS_A1",
    "STATIC_LOAD_FACTORS",
    "a_iso",
    "get_contamination_factor",
    "get_limiting_dn",
    "get_reliability_factor",
    "get_static_factors",
    "reference_viscosity",
    "viscosity_ratio",
    # Lubrication and hydrodynamic charts (Shigley ch. 12)
    "RAIMONDI_BOYD",
    "SAE_VISCOSITY_CONSTANTS",
    "is_thick_film",
    "raimondi_boyd",
    "sommerfeld_for",
    "viscosity",
    "viscosity_reyn",
    # Boundary-lubricated bushings
    "BUSHING_MATERIALS",
    "get_bushing_material",
    # Thrust pads
    "load_coefficient",
]
