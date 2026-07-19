"""Cone clutch design and analysis module.

Units convention: geometry in mm, forces in N, torques in N*mm and
pressures in MPa, consistent with the other element modules. See
:mod:`mecapy.clutches.friction_interface` for the underlying equations.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 16
(Sec. 16-7, cone clutches and brakes).
"""

import math

from .friction_interface import AxialFrictionInterface


class ConeClutch(AxialFrictionInterface):
    """
    Cone clutch — conical friction surfaces wedged by an axial force.

    The wedging action multiplies the torque of an equivalent flat disc
    by 1/sin(alpha), where alpha is the half-cone angle. Small angles
    give large torque amplification but can self-hold (grab), needing a
    pull to disengage; Shigley recommends alpha >= 8 degrees to avoid
    jamming, with 10-15 degrees typical.

    Attributes:
        cone_angle (float): Half-cone angle alpha in degrees.
    """

    def __init__(self, outer_diameter, inner_diameter, cone_angle, mu=None,
                 lining=None, material="steel", name=None):
        """
        Initialize a ConeClutch object.

        Args:
            outer_diameter (float): Friction-face outer diameter D in mm.
            inner_diameter (float): Friction-face inner diameter d in mm.
            cone_angle (float): Half-cone angle alpha in degrees, in
                (0, 45]. Values below ~8 degrees tend to jam.
            mu (float): Friction coefficient. Defaults to the lining's dry
                value when ``lining`` is given, else to the module default.
            lining (str): Optional friction-lining name from
                :mod:`mecapy.clutches.friction_data`.
            material (str): Backing material type (default: "steel").
            name (str): Optional identifier for the clutch.

        Raises:
            ValueError: If ``cone_angle`` is outside (0, 45], plus the
                checks in :class:`AxialFrictionInterface`.
        """
        if not 0 < cone_angle <= 45:
            raise ValueError("Cone angle must be in (0, 45] degrees for a cone clutch")
        super().__init__(outer_diameter, inner_diameter, mu=mu, n_faces=1,
                         cone_angle=cone_angle, lining=lining, material=material,
                         name=name)

    @property
    def face_width(self):
        """float: Slant face width b = (D - d) / (2 * sin(alpha)) in mm."""
        return (self.outer_diameter - self.inner_diameter) / (2 * self._sin_alpha)

    @property
    def is_self_holding(self):
        """bool: True when tan(alpha) < mu, i.e. the cone grabs and needs a pull to release."""
        return math.tan(math.radians(self.cone_angle)) < self.mu
