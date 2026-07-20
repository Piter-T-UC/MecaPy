"""
MecaPy - Python library for mechanical engineering calculations.

A comprehensive library for analyzing and designing mechanical elements including:
- Beams
- Wheels and flywheels
- Gears
- Bearings
- Bolts
- Welds
- Shafts
- Clutches
- Brakes
- Couplings
- Belts (flat and V)
- Roller chains
- And other mechanical components
"""

__version__ = "0.1.0"
__author__ = "Piter-T-UC"
__license__ = "MIT"

from .base import MechaElement
from .materials import Material
from . import materials
from . import beams
from . import wheels
from . import gears
from . import bearings
from . import bolts
from . import welds
from . import shafts
from . import clutches
from . import brakes
from . import couplings
from . import belts
from . import chains
from . import utils

__all__ = [
    "MechaElement",
    "Material",
    "materials",
    "beams",
    "wheels",
    "gears",
    "bearings",
    "bolts",
    "welds",
    "shafts",
    "clutches",
    "brakes",
    "couplings",
    "belts",
    "chains",
    "utils",
]
