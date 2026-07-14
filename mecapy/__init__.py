"""
MecaPy - Python library for mechanical engineering calculations.

A comprehensive library for analyzing and designing mechanical elements including:
- Beams
- Wheels
- Gears (AGMA)
- Bearings
- Bolts
- Welds
- Shafts
- Belts
- Chains
- And other mechanical components
"""

__version__ = "0.2.0"
__author__ = "Piter-T-UC"
__license__ = "MIT"

from .base import MechaElement
from .materials import Material, get_material, get_material_properties, get_available_materials
from . import beams
from . import wheels
from . import gears
from . import bearings
from . import bolts
from . import welds
from . import shafts
from . import belts
from . import chains
from . import utils

__all__ = [
    "MechaElement",
    "Material",
    "get_material",
    "get_material_properties",
    "get_available_materials",
    "beams",
    "wheels",
    "gears",
    "bearings",
    "bolts",
    "welds",
    "shafts",
    "belts",
    "chains",
    "utils",
]
