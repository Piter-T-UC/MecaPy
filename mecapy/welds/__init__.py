"""Weld design and analysis module.

- :class:`Weld` — a single weld run along a line or circle path.
- :class:`WeldLine`, :class:`WeldCircle` — weld path geometries.
- :class:`WeldedUnion` — a weld group under combined loading, analysed by
  the elastic "weld as a line" method (Shigley Ch. 9).
- :mod:`mecapy.welds.electrode_data` — electrode strength table
  (:data:`ELECTRODES`, :func:`get_electrode`).
"""

from .weld import Weld
from .geometry import WeldLine, WeldCircle, WeldPath
from .welded_union import WeldedUnion
from .electrode_data import ELECTRODES, get_electrode

__all__ = [
    "Weld",
    "WeldLine",
    "WeldCircle",
    "WeldPath",
    "WeldedUnion",
    "ELECTRODES",
    "get_electrode",
]
