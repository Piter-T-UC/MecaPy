"""Gear design and analysis subpackage.

Gear types, transmissions and AGMA fatigue rating:

- :class:`Gear` — base class (backward-compatible generic gear).
- :class:`SpurGear`, :class:`HelicalGear`, :class:`HerringboneGear` —
  parallel-axis (cylindrical) gears with AGMA rating support.
- :class:`Rack` — rack-and-pinion.
- :class:`BevelGear` — intersecting axes (simplified strength check).
- :class:`Worm`, :class:`WormWheel` — worm drives.
- :class:`PlanetaryGearSet` — epicyclic sets (Willis kinematics).
- :class:`Transmission` — multi-stage trains with compatibility checks.
- :mod:`mecapy.gears.agma` — AGMA bending/pitting rating
  (:class:`~mecapy.gears.agma.AGMARating`).
"""

from .gear import Gear
from .cylindrical import (
    CylindricalGear,
    SpurGear,
    HelicalGear,
    HerringboneGear,
)
from .rack import Rack
from .bevel import BevelGear
from .worm import Worm, WormWheel
from .planetary import PlanetaryGearSet
from .transmission import Transmission
from .agma import AGMARating
from . import agma, agma_data

__all__ = [
    "Gear",
    "CylindricalGear",
    "SpurGear",
    "HelicalGear",
    "HerringboneGear",
    "Rack",
    "BevelGear",
    "Worm",
    "WormWheel",
    "PlanetaryGearSet",
    "Transmission",
    "AGMARating",
    "agma",
    "agma_data",
]
