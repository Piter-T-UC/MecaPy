"""Shared axial friction-interface math for disc and cone clutches/brakes.

Units convention: geometry in mm, forces in N, torques in N*mm and
pressures in MPa (N/mm^2), consistent with the other element modules
(shafts, gears, bolts). Energy/temperature questions are handled by the
SI helpers in :mod:`mecapy.utils.thermal`; the only unit bridge needed
is N*mm -> N*m (divide by 1e3), done once in :meth:`power`.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 16
(Sec. 16-5 frictional-contact axial clutches, Sec. 16-7 cone clutches
and brakes). A flat disc is the ``cone_angle = 90`` degree special case
of the cone equations (sin(alpha) = 1), so one class serves both.

Vocabulary note (the classic Ch. 16 trap): ``p_max`` arguments belong
to the uniform-WEAR theory, where the pressure peaks at the inner
radius; ``pressure`` arguments belong to the uniform-PRESSURE theory,
where it is constant over the face. The two theories are exposed as
method pairs so both are always available for comparison — uniform
pressure suits new, rigid, accurately aligned surfaces; uniform wear is
the conservative worn-in assumption.
"""

import math

from ..base import MechaElement
from .friction_data import get_friction_material

DEFAULT_MU = 0.30  # dry friction coefficient (typical molded/sintered lining)


def resolve_radii(outer_radius=None, inner_radius=None,
                  outer_diameter=None, inner_diameter=None):
    """
    Resolve a friction face's radii from a radius OR diameter pair.

    Radius is the friction family's canonical geometry input; a diameter
    pair is accepted for convenience (and backward compatibility). Give
    exactly one complete pair.

    Args:
        outer_radius, inner_radius (float): Face radii in mm (canonical).
        outer_diameter, inner_diameter (float): Face diameters in mm.

    Returns:
        tuple: ``(outer_radius, inner_radius)`` in mm.

    Raises:
        ValueError: If both a radius and a diameter input are given, or a
            pair is incomplete.
    """
    have_radius = outer_radius is not None or inner_radius is not None
    have_diameter = outer_diameter is not None or inner_diameter is not None
    if have_radius and have_diameter:
        raise ValueError(
            "Give radii or diameters, not both"
        )
    if have_diameter:
        if outer_diameter is None or inner_diameter is None:
            raise ValueError("Give both outer_diameter and inner_diameter")
        return outer_diameter / 2, inner_diameter / 2
    if outer_radius is None or inner_radius is None:
        raise ValueError(
            "Give both outer_radius and inner_radius (or a diameter pair)"
        )
    return outer_radius, inner_radius


class AxialFrictionInterface(MechaElement):
    """
    Annular axial friction interface (base for disc and cone clutches).

    Models one or more annular friction faces clamped by an axial
    actuating force, with the uniform-pressure and uniform-wear
    force/torque/pressure relations. The half-cone angle ``alpha``
    generalizes the geometry: 90 degrees is a flat disc, smaller angles
    wedge the surfaces together and multiply the torque by 1/sin(alpha).

    The friction family's canonical geometry input is the **radius**;
    diameters are accepted for convenience and exposed as settable derived
    properties.

    Attributes:
        outer_radius (float): Friction-face outer radius in mm. Settable.
        inner_radius (float): Friction-face inner radius in mm. Settable.
        outer_diameter (float): Outer diameter D = 2*outer_radius in mm.
            Settable (updates the radius).
        inner_diameter (float): Inner diameter d = 2*inner_radius in mm.
            Settable.
        mu (float): Friction coefficient in use.
        n_faces (int): Number of friction faces sharing the torque.
        cone_angle (float): Half-cone angle alpha in degrees (90 = flat disc).
        lining (str): Friction-lining name, or None.
        material (str): Backing material type.
    """

    def __init__(self, outer_radius=None, inner_radius=None, mu=None,
                 n_faces=1, cone_angle=90.0, lining=None, material="steel",
                 name=None, outer_diameter=None, inner_diameter=None):
        """
        Initialize an axial friction interface.

        Args:
            outer_radius (float): Friction-face outer radius in mm
                (canonical). Give a radius pair OR a diameter pair.
            inner_radius (float): Friction-face inner radius in mm. Must be
                smaller than ``outer_radius``.
            mu (float): Friction coefficient. Defaults to the lining's dry
                value when ``lining`` is given, else to ``DEFAULT_MU``.
                An explicit value always wins over the lining default.
            n_faces (int): Number of friction faces N (default 1; a single
                plate clutch gripped on both sides has 2).
            cone_angle (float): Half-cone angle alpha in degrees, in
                (0, 90]. 90 degrees is a flat disc (default).
            lining (str): Optional friction-lining name from
                :mod:`mecapy.clutches.friction_data`; supplies the default
                ``mu`` and enables :meth:`pressure_safety_factor`.
            material (str): Backing material type (default: "steel").
            name (str): Optional identifier for the element.
            outer_diameter (float): Outer diameter D in mm — an alternative
                to ``outer_radius`` (give one geometry pair, not both).
            inner_diameter (float): Inner diameter d in mm.

        Raises:
            ValueError: For non-positive or inverted radii, both a radius
                and a diameter pair, a friction coefficient outside
                (0, 1.5), ``n_faces`` below 1, a cone angle outside
                (0, 90], or an unknown lining name.
        """
        super().__init__(name=name, material=material)
        outer_radius, inner_radius = resolve_radii(
            outer_radius, inner_radius, outer_diameter, inner_diameter)
        if not isinstance(n_faces, int) or n_faces < 1:
            raise ValueError("n_faces must be an integer >= 1")
        if not 0 < cone_angle <= 90:
            raise ValueError("Cone angle must be in (0, 90] degrees")

        self._lining_data = None
        if lining is not None:
            self._lining_data = get_friction_material(lining)
        if mu is None:
            mu = self._lining_data["mu_dry"] if self._lining_data else DEFAULT_MU
        if not 0 < mu < 1.5:
            raise ValueError("Friction coefficient must be in (0, 1.5)")

        self.outer_radius = outer_radius  # validating setters below
        self.inner_radius = inner_radius
        self.mu = mu
        self.n_faces = n_faces
        self.cone_angle = cone_angle
        self.lining = lining

    # ---- Geometry (radius canonical; diameter derived) ----

    @property
    def outer_radius(self):
        """float: Friction-face outer radius in mm."""
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, value):
        if value <= 0:
            raise ValueError("Outer radius must be strictly positive")
        if getattr(self, "_inner_radius", 0.0) >= value:
            raise ValueError("Outer radius must be larger than inner radius")
        self._outer_radius = value

    @property
    def inner_radius(self):
        """float: Friction-face inner radius in mm."""
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, value):
        if value <= 0:
            raise ValueError("Inner radius must be strictly positive")
        if value >= self._outer_radius:
            raise ValueError("Inner radius must be smaller than outer radius")
        self._inner_radius = value

    @property
    def outer_diameter(self):
        """float: Outer diameter D = 2*outer_radius in mm."""
        return 2 * self._outer_radius

    @outer_diameter.setter
    def outer_diameter(self, value):
        self.outer_radius = value / 2

    @property
    def inner_diameter(self):
        """float: Inner diameter d = 2*inner_radius in mm."""
        return 2 * self._inner_radius

    @inner_diameter.setter
    def inner_diameter(self, value):
        self.inner_radius = value / 2

    @property
    def mean_diameter(self):
        """float: Mean diameter (D + d)/2 in mm."""
        return (self.outer_diameter + self.inner_diameter) / 2

    @property
    def friction_area(self):
        """float: Friction (slant) area per face pi*(ro^2 - ri^2)/sin(alpha) in mm^2."""
        return (math.pi * (self.outer_radius ** 2 - self.inner_radius ** 2)
                / self._sin_alpha)

    @property
    def _sin_alpha(self):
        """float: sin of the half-cone angle (1.0 for a flat disc)."""
        return math.sin(math.radians(self.cone_angle))

    # ---- Uniform wear (worn-in surfaces, pressure peaks at d) ----

    def actuating_force_uniform_wear(self, p_max):
        """
        Axial actuating force that produces a given peak pressure, uniform wear.

        ::

            F = pi * p_max * d * (D - d) / 2       (Shigley Eq. 16-23)

        Args:
            p_max (float): Peak lining pressure (at the inner radius) in MPa.

        Returns:
            float: Actuating force F in N.

        Raises:
            ValueError: If ``p_max`` is not strictly positive.
        """
        if p_max <= 0:
            raise ValueError("p_max must be strictly positive")
        return (math.pi * p_max * self.inner_diameter
                * (self.outer_diameter - self.inner_diameter) / 2)

    def torque_uniform_wear(self, actuating_force):
        """
        Torque capacity for a given actuating force, uniform wear.

        ::

            T = N * mu * F * (D + d) / (4 * sin(alpha))    (Eqs. 16-25 / 16-47)

        Args:
            actuating_force (float): Axial clamping force F in N.

        Returns:
            float: Torque capacity in N*mm.

        Raises:
            ValueError: If ``actuating_force`` is not strictly positive.
        """
        if actuating_force <= 0:
            raise ValueError("Actuating force must be strictly positive")
        return (self.n_faces * self.mu * actuating_force
                * (self.outer_diameter + self.inner_diameter)
                / (4 * self._sin_alpha))

    def torque_capacity_uniform_wear(self, p_max):
        """
        Torque capacity at a given peak lining pressure, uniform wear.

        ::

            T = N * pi * mu * p_max * d * (D^2 - d^2) / (8 * sin(alpha))
                                                        (Eqs. 16-24 / 16-46)

        Args:
            p_max (float): Peak lining pressure in MPa.

        Returns:
            float: Torque capacity in N*mm.

        Raises:
            ValueError: If ``p_max`` is not strictly positive.
        """
        return self.torque_uniform_wear(self.actuating_force_uniform_wear(p_max))

    def max_pressure_uniform_wear(self, actuating_force):
        """
        Peak lining pressure for a given actuating force, uniform wear.

        Inverts Eq. 16-23; the peak occurs at the inner radius.

        Args:
            actuating_force (float): Axial clamping force F in N.

        Returns:
            float: Peak pressure at the inner radius in MPa.

        Raises:
            ValueError: If ``actuating_force`` is not strictly positive.
        """
        if actuating_force <= 0:
            raise ValueError("Actuating force must be strictly positive")
        return (2 * actuating_force
                / (math.pi * self.inner_diameter
                   * (self.outer_diameter - self.inner_diameter)))

    # ---- Uniform pressure (new, rigid, aligned surfaces) ----

    def actuating_force_uniform_pressure(self, pressure):
        """
        Axial actuating force for a given uniform face pressure.

        ::

            F = pi * p * (D^2 - d^2) / 4           (Shigley Eq. 16-27)

        Args:
            pressure (float): Uniform lining pressure p in MPa.

        Returns:
            float: Actuating force F in N.

        Raises:
            ValueError: If ``pressure`` is not strictly positive.
        """
        if pressure <= 0:
            raise ValueError("Pressure must be strictly positive")
        return (math.pi * pressure
                * (self.outer_diameter ** 2 - self.inner_diameter ** 2) / 4)

    def torque_uniform_pressure(self, actuating_force):
        """
        Torque capacity for a given actuating force, uniform pressure.

        ::

            T = N * mu * F * (D^3 - d^3) / (3 * sin(alpha) * (D^2 - d^2))
                                                        (Eqs. 16-28 / 16-49)

        Args:
            actuating_force (float): Axial clamping force F in N.

        Returns:
            float: Torque capacity in N*mm.

        Raises:
            ValueError: If ``actuating_force`` is not strictly positive.
        """
        if actuating_force <= 0:
            raise ValueError("Actuating force must be strictly positive")
        d3 = self.outer_diameter ** 3 - self.inner_diameter ** 3
        d2 = self.outer_diameter ** 2 - self.inner_diameter ** 2
        return self.n_faces * self.mu * actuating_force * d3 / (3 * self._sin_alpha * d2)

    def pressure_uniform_pressure(self, actuating_force):
        """
        Uniform face pressure for a given actuating force.

        Inverts Eq. 16-27.

        Args:
            actuating_force (float): Axial clamping force F in N.

        Returns:
            float: Uniform lining pressure in MPa.

        Raises:
            ValueError: If ``actuating_force`` is not strictly positive.
        """
        if actuating_force <= 0:
            raise ValueError("Actuating force must be strictly positive")
        return (4 * actuating_force
                / (math.pi * (self.outer_diameter ** 2 - self.inner_diameter ** 2)))

    # ---- Lining checks and power ----

    def pressure_safety_factor(self, actuating_force, theory="uniform_wear",
                               p_allowable=None):
        """
        Ratio of the lining's allowable pressure to the working pressure.

        Args:
            actuating_force (float): Axial clamping force F in N.
            theory (str): "uniform_wear" (peak pressure at the inner
                radius, conservative, default) or "uniform_pressure".
            p_allowable (float): Allowable pressure in MPa. Defaults to
                the lining's ``p_max`` when a lining was given.

        Returns:
            float: p_allowable / p_working. Values greater than 1 mean the
            lining is within its pressure rating.

        Raises:
            ValueError: If no allowable pressure is available (no lining
                and no ``p_allowable``), or ``theory`` is unknown.
        """
        if p_allowable is None:
            if self._lining_data is None:
                raise ValueError(
                    "No allowable pressure: give p_allowable or construct "
                    "the element with a lining"
                )
            p_allowable = self._lining_data["p_max"]
        if theory == "uniform_wear":
            p_working = self.max_pressure_uniform_wear(actuating_force)
        elif theory == "uniform_pressure":
            p_working = self.pressure_uniform_pressure(actuating_force)
        else:
            raise ValueError("theory must be 'uniform_wear' or 'uniform_pressure'")
        return p_allowable / p_working

    def power(self, actuating_force, speed_rpm, theory="uniform_wear"):
        """
        Power transmitted at a given speed, fully engaged (no slip).

        Converts the torque from N*mm to N*m (divide by 1e3) and the
        speed from rpm to rad/s, then applies ``H = T * w``.

        Args:
            actuating_force (float): Axial clamping force F in N.
            speed_rpm (float): Shaft speed in rev/min.
            theory (str): "uniform_wear" (default) or "uniform_pressure".

        Returns:
            float: Transmitted power in W.

        Raises:
            ValueError: If ``speed_rpm`` is not strictly positive or
                ``theory`` is unknown.
        """
        if speed_rpm <= 0:
            raise ValueError("Speed must be strictly positive")
        if theory == "uniform_wear":
            torque = self.torque_uniform_wear(actuating_force)
        elif theory == "uniform_pressure":
            torque = self.torque_uniform_pressure(actuating_force)
        else:
            raise ValueError("theory must be 'uniform_wear' or 'uniform_pressure'")
        omega = speed_rpm * 2 * math.pi / 60
        return torque / 1e3 * omega
