"""Clutch design and analysis module."""

from .friction_interface import AxialFrictionInterface
from .disc import DiscClutch
from .cone import ConeClutch
from .centrifugal import CentrifugalClutch
from .friction_data import FRICTION_MATERIALS, get_friction_material

__all__ = [
    "AxialFrictionInterface",
    "DiscClutch",
    "ConeClutch",
    "CentrifugalClutch",
    "FRICTION_MATERIALS",
    "get_friction_material",
]
