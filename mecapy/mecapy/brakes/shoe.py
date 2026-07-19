"""Pivoted long-shoe drum brake design and analysis module.

Units convention: geometry in mm, forces in N, torques in N*mm and
pressures in MPa, consistent with the other element modules. Angles are
given in degrees and converted internally.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 16
(Sec. 16-3 internal expanding shoes, Eqs. 16-2 to 16-8; Sec. 16-4
external contracting shoes, Eqs. 16-9 to 16-13; Fig. 16-7 geometry).

Frame convention (Shigley Fig. 16-7): the shoe angle ``theta`` is
measured at the DRUM CENTER from the line through the drum center and
the hinge pin; ``theta1``/``theta2`` locate the shoe heel and toe on
that scale. The pressure distribution is ``p = p_max * sin(theta) /
sin(theta_a)`` with ``theta_a = min(theta2, 90 deg)`` the angle of peak
pressure. Hinge-pin reactions are returned in the same frame: x-axis
along the drum-center/hinge-pin line, y-axis toward the shoe.

Whether the shoe is self-energizing (drum friction helps apply the
shoe) or de-energizing depends on the drum rotation direction relative
to the shoe layout; here that physical choice is passed directly as
``rotation="self_energizing"`` or ``"de_energizing"``, which selects
the sign with which the friction moment enters the actuating-force and
hinge-reaction equations. Internal and external shoes share the same
mechanics — the classes differ only in documentation of the layout.
"""

import math

from ..base import MechaElement
from ..clutches.friction_data import get_friction_material
from ..clutches.friction_interface import DEFAULT_MU

_ROTATIONS = ("self_energizing", "de_energizing")


class _PivotedShoeBrake(MechaElement):
    """
    Shared mechanics of a pivoted long-shoe drum brake.

    Every quantity is linear in the peak pressure ``p_max``, so each
    public method has a per-unit-pressure private form that also allows
    clean inversion (finding the pressure from a given actuating force).

    Attributes:
        drum_radius (float): Drum friction-surface radius r in mm.
        face_width (float): Shoe (lining) face width b in mm.
        pivot_distance (float): Distance a from drum center to hinge pin in mm.
        theta1 (float): Heel angle of the lining in degrees (from the
            drum-center/hinge-pin line).
        theta2 (float): Toe angle of the lining in degrees.
        actuation_arm (float): Perpendicular distance c from the hinge pin
            to the actuating-force line in mm.
        mu (float): Friction coefficient between lining and drum.
        rotation (str): "self_energizing" or "de_energizing".
        lining (str): Friction-lining name, or None.
    """

    def __init__(self, drum_radius, face_width, pivot_distance, theta1, theta2,
                 actuation_arm, mu=None, rotation="self_energizing",
                 lining=None, material="steel", name=None):
        """
        Initialize a pivoted-shoe brake.

        Args:
            drum_radius (float): Drum friction-surface radius r in mm.
            face_width (float): Shoe face width b in mm.
            pivot_distance (float): Distance a from the drum center to the
                hinge pin in mm.
            theta1 (float): Heel angle of the lining in degrees. May be 0
                (lining starts on the hinge-pin line).
            theta2 (float): Toe angle of the lining in degrees. Must
                satisfy ``0 <= theta1 < theta2 <= 180``.
            actuation_arm (float): Perpendicular distance c from the hinge
                pin to the actuating-force line in mm.
            mu (float): Friction coefficient. Defaults to the lining's dry
                value when ``lining`` is given, else to the clutch-module
                default.
            rotation (str): "self_energizing" (drum friction helps apply
                the shoe, default) or "de_energizing".
            lining (str): Optional friction-lining name from
                :mod:`mecapy.clutches.friction_data`.
            material (str): Shoe material type (default: "steel").
            name (str): Optional identifier for the brake.

        Raises:
            ValueError: For non-positive lengths, angles outside
                ``0 <= theta1 < theta2 <= 180``, an unknown ``rotation``
                string, a friction coefficient outside (0, 1.5), or an
                unknown lining name.
        """
        super().__init__(name=name, material=material)
        if drum_radius <= 0:
            raise ValueError("Drum radius must be strictly positive")
        if face_width <= 0:
            raise ValueError("Face width must be strictly positive")
        if pivot_distance <= 0:
            raise ValueError("Pivot distance must be strictly positive")
        if actuation_arm <= 0:
            raise ValueError("Actuation arm must be strictly positive")
        if not 0 <= theta1 < theta2 <= 180:
            raise ValueError("Angles must satisfy 0 <= theta1 < theta2 <= 180 degrees")
        if rotation not in _ROTATIONS:
            raise ValueError("rotation must be 'self_energizing' or 'de_energizing'")

        self._lining_data = None
        if lining is not None:
            self._lining_data = get_friction_material(lining)
        if mu is None:
            mu = self._lining_data["mu_dry"] if self._lining_data else DEFAULT_MU
        if not 0 < mu < 1.5:
            raise ValueError("Friction coefficient must be in (0, 1.5)")

        self.drum_radius = drum_radius
        self.face_width = face_width
        self.pivot_distance = pivot_distance
        self.theta1 = theta1
        self.theta2 = theta2
        self.actuation_arm = actuation_arm
        self.mu = mu
        self.rotation = rotation
        self.lining = lining

    # ---- Geometry ----

    @property
    def theta_a(self):
        """float: Angle of peak pressure theta_a = min(theta2, 90) in degrees."""
        return min(self.theta2, 90.0)

    @property
    def _sin_theta_a(self):
        """float: sin(theta_a)."""
        return math.sin(math.radians(self.theta_a))

    @property
    def _int_sin(self):
        """float: Integral of sin(theta) over the lining, cos(theta1) - cos(theta2)."""
        t1, t2 = math.radians(self.theta1), math.radians(self.theta2)
        return math.cos(t1) - math.cos(t2)

    @property
    def _int_sin2(self):
        """float: Integral of sin^2(theta) over the lining (Shigley's B)."""
        t1, t2 = math.radians(self.theta1), math.radians(self.theta2)
        return (t2 - t1) / 2 - (math.sin(2 * t2) - math.sin(2 * t1)) / 4

    @property
    def _int_sin_cos(self):
        """float: Integral of sin(theta)*cos(theta) over the lining (Shigley's A)."""
        t1, t2 = math.radians(self.theta1), math.radians(self.theta2)
        return (math.sin(t2) ** 2 - math.sin(t1) ** 2) / 2

    # ---- Per-unit-pressure forms (everything is linear in p_max) ----

    @property
    def _moment_friction_per_pa(self):
        """float: Friction-force moment about the hinge pin per unit p_max, N*mm/MPa."""
        r, a = self.drum_radius, self.pivot_distance
        return (self.mu * self.face_width * r / self._sin_theta_a
                * (r * self._int_sin - a * self._int_sin_cos))

    @property
    def _moment_normal_per_pa(self):
        """float: Normal-force moment about the hinge pin per unit p_max, N*mm/MPa."""
        return (self.face_width * self.drum_radius * self.pivot_distance
                * self._int_sin2 / self._sin_theta_a)

    @property
    def _torque_per_pa(self):
        """float: Braking torque per unit p_max, N*mm/MPa."""
        return (self.mu * self.face_width * self.drum_radius ** 2
                * self._int_sin / self._sin_theta_a)

    @property
    def _force_per_pa(self):
        """float: Actuating force per unit p_max, N/MPa (signed; <= 0 means self-locking)."""
        if self.rotation == "self_energizing":
            moment = self._moment_normal_per_pa - self._moment_friction_per_pa
        else:
            moment = self._moment_normal_per_pa + self._moment_friction_per_pa
        return moment / self.actuation_arm

    # ---- Moments, force and torque ----

    def moment_friction(self, p_max):
        """
        Moment of the friction forces about the hinge pin.

        ::

            Mf = (mu * pa * b * r / sin(theta_a))
                 * integral[ sin(theta) * (r - a*cos(theta)) ]     (Eq. 16-2)

        Args:
            p_max (float): Peak lining pressure in MPa.

        Returns:
            float: Friction moment Mf in N*mm.

        Raises:
            ValueError: If ``p_max`` is not strictly positive.
        """
        self._check_pressure(p_max)
        return self._moment_friction_per_pa * p_max

    def moment_normal(self, p_max):
        """
        Moment of the normal forces about the hinge pin.

        ::

            MN = (pa * b * r * a / sin(theta_a)) * integral[ sin^2(theta) ]
                                                                  (Eq. 16-3)

        Args:
            p_max (float): Peak lining pressure in MPa.

        Returns:
            float: Normal moment MN in N*mm.

        Raises:
            ValueError: If ``p_max`` is not strictly positive.
        """
        self._check_pressure(p_max)
        return self._moment_normal_per_pa * p_max

    def actuating_force(self, p_max):
        """
        Actuating force needed to reach a given peak lining pressure.

        ::

            F = (MN - Mf) / c    (self-energizing)          (Eq. 16-4)
            F = (MN + Mf) / c    (de-energizing)

        Args:
            p_max (float): Peak lining pressure in MPa.

        Returns:
            float: Actuating force F in N.

        Raises:
            ValueError: If ``p_max`` is not strictly positive, or the
                shoe is self-locking (the friction moment alone applies
                the shoe, so no positive force is defined).
        """
        self._check_pressure(p_max)
        force = self._force_per_pa * p_max
        if force <= 0:
            raise ValueError(
                "Shoe is self-locking (Mf >= MN): no actuating force is "
                "defined; reduce mu or move the hinge pin"
            )
        return force

    def torque(self, p_max):
        """
        Braking torque on the drum at a given peak lining pressure.

        ::

            T = mu * pa * b * r^2 * (cos(theta1) - cos(theta2)) / sin(theta_a)
                                                                  (Eq. 16-6)

        Args:
            p_max (float): Peak lining pressure in MPa.

        Returns:
            float: Braking torque in N*mm.

        Raises:
            ValueError: If ``p_max`` is not strictly positive.
        """
        self._check_pressure(p_max)
        return self._torque_per_pa * p_max

    def max_pressure_for_force(self, actuating_force):
        """
        Peak lining pressure produced by a given actuating force.

        Inverts :meth:`actuating_force` (everything is linear in the
        pressure). This is how the de-energizing shoe of a two-shoe
        brake is solved: both shoes share the same force, so the
        de-energizing one runs at a lower pressure.

        Args:
            actuating_force (float): Actuating force F in N.

        Returns:
            float: Peak lining pressure in MPa.

        Raises:
            ValueError: If ``actuating_force`` is not strictly positive,
                or the shoe is self-locking.
        """
        if actuating_force <= 0:
            raise ValueError("Actuating force must be strictly positive")
        force_per_pa = self._force_per_pa
        if force_per_pa <= 0:
            raise ValueError(
                "Shoe is self-locking (Mf >= MN): pressure is not defined "
                "by the actuating force"
            )
        return actuating_force / force_per_pa

    def torque_for_force(self, actuating_force):
        """
        Braking torque produced by a given actuating force.

        Composes :meth:`max_pressure_for_force` and :meth:`torque`.

        Args:
            actuating_force (float): Actuating force F in N.

        Returns:
            float: Braking torque in N*mm.

        Raises:
            ValueError: See :meth:`max_pressure_for_force`.
        """
        return self.torque(self.max_pressure_for_force(actuating_force))

    # ---- Self-energizing / self-locking ----

    @property
    def is_self_energizing(self):
        """bool: True when the drum friction helps apply the shoe."""
        return self.rotation == "self_energizing"

    @property
    def is_self_locking(self):
        """bool: True when Mf >= MN for a self-energizing shoe (grabs with no force)."""
        if not self.is_self_energizing:
            return False
        return self._moment_friction_per_pa >= self._moment_normal_per_pa

    @property
    def self_locking_margin(self):
        """float: Ratio MN/Mf (pressure-independent); values > 1 mean no self-locking."""
        return self._moment_normal_per_pa / self._moment_friction_per_pa

    # ---- Hinge-pin reactions ----

    def hinge_reactions(self, p_max, force_x=0.0, force_y=0.0):
        """
        Hinge-pin reaction components at a given peak lining pressure.

        With ``A`` and ``B`` the sin*cos and sin^2 lining integrals and
        ``k = pa*b*r/sin(theta_a)`` (Shigley Eqs. 16-7/16-8)::

            Rx = k * (A - mu*B) - Fx     (self-energizing)
            Ry = k * (B + mu*A) - Fy
            Rx = k * (A + mu*B) - Fx     (de-energizing)
            Ry = k * (B - mu*A) - Fy

        The frame is the Fig. 16-7 shoe frame documented in the module
        docstring; ``force_x``/``force_y`` are the actuating-force
        components expressed in that same frame.

        Args:
            p_max (float): Peak lining pressure in MPa.
            force_x (float): Actuating-force x-component in N (default 0).
            force_y (float): Actuating-force y-component in N (default 0).

        Returns:
            tuple: ``(Rx, Ry)`` hinge-pin reaction components in N.

        Raises:
            ValueError: If ``p_max`` is not strictly positive.
        """
        self._check_pressure(p_max)
        k = p_max * self.face_width * self.drum_radius / self._sin_theta_a
        a_int, b_int = self._int_sin_cos, self._int_sin2
        if self.is_self_energizing:
            rx = k * (a_int - self.mu * b_int) - force_x
            ry = k * (b_int + self.mu * a_int) - force_y
        else:
            rx = k * (a_int + self.mu * b_int) - force_x
            ry = k * (b_int - self.mu * a_int) - force_y
        return rx, ry

    # ---- Internal helpers ----

    @staticmethod
    def _check_pressure(p_max):
        """Validate a peak lining pressure."""
        if p_max <= 0:
            raise ValueError("p_max must be strictly positive")


class InternalShoeBrake(_PivotedShoeBrake):
    """
    Internal-expanding pivoted-shoe (drum) brake.

    The shoe presses outward against the inside of the drum — the
    common automotive drum-brake layout (Shigley Sec. 16-3, Fig. 16-7).
    A two-shoe drum brake is modeled as two instances sharing the same
    actuating force: one ``rotation="self_energizing"`` (primary shoe)
    and one ``"de_energizing"`` (secondary shoe, solved with
    :meth:`torque_for_force` at the shared force).

    See :class:`_PivotedShoeBrake` for the mechanics and the frame
    convention.
    """


class ExternalShoeBrake(_PivotedShoeBrake):
    """
    External-contracting pivoted-shoe brake.

    The shoe presses inward against the outside of the drum (Shigley
    Sec. 16-4, Eqs. 16-9 to 16-13). The mechanics mirror the internal
    shoe exactly; which drum rotation direction makes the shoe
    self-energizing is simply reversed, and that physical choice is
    already expressed by the ``rotation`` argument.

    See :class:`_PivotedShoeBrake` for the mechanics and the frame
    convention.
    """
