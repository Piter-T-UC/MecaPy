"""Brake design and analysis module."""

from .shoe import InternalShoeBrake, ExternalShoeBrake
from .band import BandBrake
from .disc import CaliperDiscBrake

# A full-annulus disc brake is a disc clutch used to stop the load —
# same geometry, same uniform-pressure/uniform-wear equations — so the
# clutch class is re-exported under the brake name rather than duplicated.
from ..clutches.disc import DiscClutch as DiscBrake

__all__ = [
    "InternalShoeBrake",
    "ExternalShoeBrake",
    "BandBrake",
    "CaliperDiscBrake",
    "DiscBrake",
]
