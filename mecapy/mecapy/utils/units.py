"""Optional pint integration helpers.

Every mecaPy class keeps working with plain floats exactly as before.
These helpers let a class *also* accept a ``pint.Quantity`` at its public
boundary (constructor arguments, property setters), converting it to the
class's documented unit and unwrapping it to a plain float. Internal
storage and all derived ``@property`` values remain plain floats -- pint
is only ever at the boundary, never threaded through internal
computation.

``pint`` is an optional dependency (``pip install mecapy[units]``). When
it is not installed, :func:`to_magnitude` simply coerces to ``float`` and
every existing plain-float caller behaves identically.
"""

try:
    import pint

    ureg = pint.UnitRegistry()
except ImportError:  # pint is optional; plain floats always work without it
    pint = None
    ureg = None


def to_magnitude(value, unit):
    """
    Coerce a value to a plain float in the given unit.

    Args:
        value: Either a plain number (assumed already expressed in
            ``unit``, for full backward compatibility) or a
            ``pint.Quantity`` (converted to ``unit``).
        unit (str): Target unit string understood by pint, e.g. "mm",
            "N", "MPa".

    Returns:
        float: Magnitude in ``unit``.

    Raises:
        pint.DimensionalityError: If ``value`` is a Quantity with
            incompatible dimensions.
    """
    if ureg is not None and isinstance(value, ureg.Quantity):
        return value.to(unit).magnitude
    return float(value)
