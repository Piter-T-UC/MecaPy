"""Disc (plate) clutch design and analysis module.

Units convention: geometry in mm, forces in N, torques in N*mm and
pressures in MPa, consistent with the other element modules. See
:mod:`mecapy.clutches.friction_interface` for the underlying equations.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 16
(Sec. 16-5, frictional-contact axial clutches).
"""

import math

from .friction_interface import AxialFrictionInterface


class DiscClutch(AxialFrictionInterface):
    """
    Flat annular disc (plate) clutch — or full-annulus disc brake.

    The flat-disc special case of :class:`AxialFrictionInterface`
    (half-cone angle fixed at 90 degrees). A disc brake with a
    full-annulus lining is the same element used to stop rather than
    couple, so this class doubles as one; the pad-sector caliper brake
    lives in :class:`mecapy.brakes.CaliperDiscBrake`.

    A multi-plate clutch is modeled with ``n_faces`` equal to the number
    of friction surfaces (a single plate gripped on both sides has 2).
    """

    def __init__(self, outer_radius=None, inner_radius=None, mu=None,
                 n_faces=1, lining=None, material="steel", name=None,
                 outer_diameter=None, inner_diameter=None):
        """
        Initialize a DiscClutch object.

        Args:
            outer_radius (float): Friction-face outer radius in mm
                (canonical). Give a radius pair OR a diameter pair.
            inner_radius (float): Friction-face inner radius in mm.
            mu (float): Friction coefficient. Defaults to the lining's dry
                value when ``lining`` is given, else to the module default.
            n_faces (int): Number of friction faces N (default 1).
            lining (str): Optional friction-lining name from
                :mod:`mecapy.clutches.friction_data`.
            material (str): Backing material type (default: "steel").
            name (str): Optional identifier for the clutch.
            outer_diameter (float): Outer diameter D in mm (alternative to
                ``outer_radius``).
            inner_diameter (float): Inner diameter d in mm.

        Raises:
            ValueError: See :class:`AxialFrictionInterface`.
        """
        super().__init__(outer_radius=outer_radius, inner_radius=inner_radius,
                         mu=mu, n_faces=n_faces, cone_angle=90.0,
                         lining=lining, material=material, name=name,
                         outer_diameter=outer_diameter,
                         inner_diameter=inner_diameter)

    @property
    def optimal_inner_diameter(self):
        """float: Inner diameter d = D/sqrt(3) in mm maximizing uniform-wear torque."""
        return self.outer_diameter / math.sqrt(3)

    def is_optimal_ratio(self, tolerance=0.05):
        """
        Check whether the diameter ratio is near the uniform-wear optimum.

        The uniform-wear torque at fixed peak pressure is maximized when
        d = D/sqrt(3), i.e. d/D ~ 0.577.

        Args:
            tolerance (float): Allowed relative deviation of the inner
                diameter from the optimum (default 0.05, i.e. 5%).

        Returns:
            bool: True when the inner diameter is within the tolerance of
            the optimum.
        """
        return (abs(self.inner_diameter - self.optimal_inner_diameter)
                / self.optimal_inner_diameter <= tolerance)
