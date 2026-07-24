"""Power screw (lead screw) design and analysis module.

Units convention: geometry in mm, forces in N, torques in N*mm and
stresses in MPa (N/mm^2), consistent with the other element modules
(shafts, gears, bolts). Thread-form geometry factors come from
:mod:`mecapy.shafts.power_screw_data`.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 8 (power
screws). The sec(alpha) half-angle correction in the torque equations
is the standard axial-plane approximation.
"""

import math

from ..base import MechaElement
from ..columns import Column
from .power_screw_data import get_thread_form

DEFAULT_MU = 0.15         # thread friction coefficient (steel on steel, greasy)
DEFAULT_MU_COLLAR = 0.15  # collar (thrust) friction coefficient


class PowerScrew(MechaElement):
    """
    Power screw (lead screw) design and analysis.

    Models a translating screw that raises or lowers an axial load,
    covering the raising/lowering torque (with optional collar friction),
    efficiency, self-locking, the screw-body stresses, the thread
    engagement stresses and a column-buckling check.

    Thread geometry uses the mean (pitch) diameter ``dm = d - p/2`` and
    the root (minor) diameter ``dr = d - p`` of an idealized square/Acme
    thread. The thread half-angle (0 for square, 14.5 deg for Acme, ...)
    enters the torque equations through a sec(alpha) correction.

    Attributes:
        major_diameter (float): Nominal (major) diameter d in mm.
        pitch (float): Single-thread pitch p in mm.
        length (float): Screw length in mm (used for buckling).
        n_starts (int): Number of thread starts.
        thread_form (str): Thread form name, or "custom".
        material (str): Material type.
    """

    def __init__(self, major_diameter, pitch, length, n_starts=1,
                 thread_form="square", thread_angle=None, material="steel", name=None):
        """
        Initialize a PowerScrew object.

        Args:
            major_diameter (float): Nominal (major) diameter d in mm.
            pitch (float): Single-thread pitch p in mm.
            length (float): Screw length in mm.
            n_starts (int): Number of thread starts (default: 1).
            thread_form (str): One of "square", "acme", "trapezoidal",
                "buttress", or "custom" (default: "square").
            thread_angle (float): Thread HALF-angle in degrees. Required
                when ``thread_form="custom"``; ignored for named forms.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the screw.

        Raises:
            ValueError: For non-positive dimensions, ``n_starts`` below 1,
                a pitch too large for a positive root diameter, an unknown
                thread form, or a missing/out-of-range custom angle.
        """
        super().__init__(name=name, material=material)
        if major_diameter <= 0:
            raise ValueError("Major diameter must be strictly positive")
        if pitch <= 0:
            raise ValueError("Pitch must be strictly positive")
        if length <= 0:
            raise ValueError("Length must be strictly positive")
        if not isinstance(n_starts, int) or n_starts < 1:
            raise ValueError("n_starts must be an integer >= 1")
        if pitch >= major_diameter:
            raise ValueError("Pitch too large: root diameter would be non-positive")

        if thread_form == "custom":
            if thread_angle is None:
                raise ValueError("thread_angle is required when thread_form='custom'")
            if not 0 <= thread_angle < 90:
                raise ValueError("thread_angle must be in the range [0, 90) degrees")
            self._thread_angle = float(thread_angle)
            self._height_factor = 0.5
        else:
            data = get_thread_form(thread_form)
            self._thread_angle = data["thread_angle_deg"]
            self._height_factor = data["thread_height_factor"]

        self.major_diameter = major_diameter
        self.pitch = pitch
        self.length = length
        self.n_starts = n_starts
        self.thread_form = thread_form

    # ---- Geometry ----

    @property
    def lead(self):
        """float: Lead l = n_starts * pitch in mm (axial travel per turn)."""
        return self.n_starts * self.pitch

    @property
    def mean_diameter(self):
        """float: Mean (pitch) diameter dm = d - p/2 in mm."""
        return self.major_diameter - self.pitch / 2

    @property
    def minor_diameter(self):
        """float: Root (minor) diameter dr = d - p in mm."""
        return self.major_diameter - self.pitch

    @property
    def lead_angle(self):
        """float: Lead angle lambda = atan(l / (pi*dm)) in degrees."""
        return math.degrees(math.atan(self.lead / (math.pi * self.mean_diameter)))

    @property
    def thread_angle(self):
        """float: Thread half-angle alpha in degrees (0 for square)."""
        return self._thread_angle

    @property
    def root_area(self):
        """float: Root cross-sectional area pi*dr^2/4 in mm^2."""
        return math.pi * self.minor_diameter ** 2 / 4

    @property
    def polar_moment(self):
        """float: Polar second moment of the root section pi*dr^4/32 (mm^4)."""
        return math.pi * self.minor_diameter ** 4 / 32

    @property
    def second_moment(self):
        """float: Second moment of area of the root section pi*dr^4/64 (mm^4)."""
        return math.pi * self.minor_diameter ** 4 / 64

    @property
    def _sec_alpha(self):
        """float: Secant of the thread half-angle (1 for a square thread)."""
        return 1.0 / math.cos(math.radians(self._thread_angle))

    # ---- Torque, efficiency, self-locking ----

    def raise_torque(self, force, mu=DEFAULT_MU, collar_diameter=None,
                     mu_collar=DEFAULT_MU_COLLAR):
        """
        Torque required to raise (lift) an axial load.

        Shigley Eq. 8-5::

            T_R = (F*dm/2) * (l + pi*mu*dm*sec a) / (pi*dm - mu*l*sec a)
                  + F*mu_c*dc/2

        Args:
            force (float): Axial load F in N.
            mu (float): Thread friction coefficient (default: 0.15).
            collar_diameter (float): Mean collar (thrust bearing) diameter
                dc in mm. If None, the collar term is omitted.
            mu_collar (float): Collar friction coefficient (default: 0.15).

        Returns:
            float: Raising torque in N*mm.

        Raises:
            ValueError: If a friction coefficient is negative.
        """
        self._check_friction(mu, mu_collar, collar_diameter)
        dm = self.mean_diameter
        sec_a = self._sec_alpha
        thread = (force * dm / 2) * (self.lead + math.pi * mu * dm * sec_a) / (
            math.pi * dm - mu * self.lead * sec_a
        )
        return thread + self._collar_torque(force, collar_diameter, mu_collar)

    def lower_torque(self, force, mu=DEFAULT_MU, collar_diameter=None,
                     mu_collar=DEFAULT_MU_COLLAR):
        """
        Torque required to lower an axial load.

        Shigley Eq. 8-6::

            T_L = (F*dm/2) * (pi*mu*dm*sec a - l) / (pi*dm + mu*l*sec a)
                  + F*mu_c*dc/2

        The thread term is negative for a non-self-locking screw (the load
        would back-drive on its own); the signed value is returned rather
        than clamped.

        Args:
            force (float): Axial load F in N.
            mu (float): Thread friction coefficient (default: 0.15).
            collar_diameter (float): Mean collar diameter dc in mm, or None.
            mu_collar (float): Collar friction coefficient (default: 0.15).

        Returns:
            float: Lowering torque in N*mm (may be negative).

        Raises:
            ValueError: If a friction coefficient is negative.
        """
        self._check_friction(mu, mu_collar, collar_diameter)
        dm = self.mean_diameter
        sec_a = self._sec_alpha
        thread = (force * dm / 2) * (math.pi * mu * dm * sec_a - self.lead) / (
            math.pi * dm + mu * self.lead * sec_a
        )
        return thread + self._collar_torque(force, collar_diameter, mu_collar)

    def efficiency(self, force, mu=DEFAULT_MU, collar_diameter=None,
                   mu_collar=DEFAULT_MU_COLLAR):
        """
        Mechanical efficiency of the screw when raising a load.

        Shigley Eq. 8-4, ``e = F*l / (2*pi*T_R)``. With ``collar_diameter``
        omitted this is the thread-only efficiency; passing a collar gives
        the overall efficiency including collar friction.

        Args:
            force (float): Axial load F in N.
            mu (float): Thread friction coefficient (default: 0.15).
            collar_diameter (float): Mean collar diameter dc in mm, or None.
            mu_collar (float): Collar friction coefficient (default: 0.15).

        Returns:
            float: Efficiency between 0 and 1.

        Raises:
            ValueError: If ``force`` is zero.
        """
        if force == 0:
            raise ValueError("Force must be non-zero to compute efficiency")
        t_r = self.raise_torque(force, mu, collar_diameter, mu_collar)
        return force * self.lead / (2 * math.pi * t_r)

    def is_self_locking(self, mu=DEFAULT_MU):
        """
        Whether the screw is self-locking (holds the load without torque).

        The thread alone is self-locking when the lowering thread torque is
        non-negative, i.e. ``mu * sec a >= tan(lambda)`` (collar friction is
        ignored — a collar only adds to the holding capacity).

        Args:
            mu (float): Thread friction coefficient (default: 0.15).

        Returns:
            bool: True if the screw is self-locking.

        Raises:
            ValueError: If ``mu`` is negative.
        """
        if mu < 0:
            raise ValueError("Friction coefficient must be non-negative")
        return mu * self._sec_alpha >= math.tan(math.radians(self.lead_angle))

    # ---- Screw-body stresses ----

    def axial_stress(self, force):
        """
        Direct axial stress on the root section.

        Args:
            force (float): Axial load F in N (tension positive).

        Returns:
            float: Axial stress 4F/(pi*dr^2) in MPa.
        """
        return 4 * force / (math.pi * self.minor_diameter ** 2)

    def torsional_stress(self, torque):
        """
        Maximum torsional shear stress on the root section.

        Uses tau = T * r / J with r = dr/2 and J = pi*dr^4/32, matching
        :meth:`mecapy.shafts.shaft.Shaft.torsional_stress` applied to the
        root diameter.

        Args:
            torque (float): Applied torque in N*mm.

        Returns:
            float: Surface shear stress in MPa.
        """
        return torque * (self.minor_diameter / 2) / self.polar_moment

    def von_mises_stress(self, force, torque):
        """
        Von Mises equivalent stress on the screw body.

        Combines the axial stress and the torsional shear stress with
        sigma' = sqrt(sigma^2 + 3*tau^2). The torque passed is normally
        the thread-only :meth:`raise_torque` (the collar torque is reacted
        at the thrust bearing, not carried by the screw body).

        Args:
            force (float): Axial load F in N.
            torque (float): Torque carried by the screw body in N*mm.

        Returns:
            float: Equivalent stress in MPa.
        """
        sigma = self.axial_stress(force)
        tau = self.torsional_stress(torque)
        return math.sqrt(sigma ** 2 + 3 * tau ** 2)

    # ---- Thread-engagement stresses ----

    def _engaged_threads(self, engagement_length):
        """Return the number of engaged threads n_e = H / p."""
        if engagement_length <= 0:
            raise ValueError("Engagement length must be strictly positive")
        return engagement_length / self.pitch

    def bearing_pressure(self, force, engagement_length):
        """
        Bearing (contact) pressure on the engaged thread flanks.

        ``sigma_B = F / (pi * dm * h * n_e)`` with the engaged thread
        height ``h = thread_height_factor * pitch`` and the number of
        engaged threads ``n_e = engagement_length / pitch``.

        Args:
            force (float): Axial load F in N.
            engagement_length (float): Nut engagement length H in mm.

        Returns:
            float: Bearing pressure in MPa.

        Raises:
            ValueError: If ``engagement_length`` is not positive.
        """
        n_e = self._engaged_threads(engagement_length)
        h = self._height_factor * self.pitch
        return force / (math.pi * self.mean_diameter * h * n_e)

    def thread_bending_stress(self, force, engagement_length, side="screw"):
        """
        Bending stress at the root of the engaged threads.

        Models the thread as a cantilever loaded at its mid-height,
        ``sigma_b = 6F / (pi * D * n_e * p)`` with ``D = dr`` on the screw
        side and ``D = major_diameter`` on the nut side. Exact for a square
        thread; approximate for other forms.

        Args:
            force (float): Axial load F in N.
            engagement_length (float): Nut engagement length H in mm.
            side (str): "screw" (root diameter) or "nut" (major diameter).

        Returns:
            float: Thread bending stress in MPa.

        Raises:
            ValueError: If ``side`` is invalid or ``engagement_length`` is
                not positive.
        """
        n_e = self._engaged_threads(engagement_length)
        diameter = self._side_diameter(side)
        return 6 * force / (math.pi * diameter * n_e * self.pitch)

    def thread_shear_stress(self, force, engagement_length, side="screw"):
        """
        Transverse shear stress at the root of the engaged threads.

        Maximum shear with the 3/2 rectangular factor,
        ``tau = 3F / (pi * D * n_e * p)`` with ``D = dr`` on the screw side
        and ``D = major_diameter`` on the nut side.

        Args:
            force (float): Axial load F in N.
            engagement_length (float): Nut engagement length H in mm.
            side (str): "screw" (root diameter) or "nut" (major diameter).

        Returns:
            float: Thread shear stress in MPa.

        Raises:
            ValueError: If ``side`` is invalid or ``engagement_length`` is
                not positive.
        """
        n_e = self._engaged_threads(engagement_length)
        diameter = self._side_diameter(side)
        return 3 * force / (math.pi * diameter * n_e * self.pitch)

    # ---- Buckling and safety ----

    @property
    def elastic_modulus(self):
        """float: Elastic modulus of the screw material in MPa."""
        return self.material_properties["elastic_modulus"] / 1e6

    def buckling_column(self, end_condition=1.0):
        """
        The screw root modelled as a :class:`~mecapy.columns.Column`.

        Builds a solid-circular column on the root (minor) diameter, so
        the buckling checks (Euler/Johnson slenderness and loads) are
        shared with the general column element rather than duplicated.

        Args:
            end_condition (float): Effective-length factor K (default 1.0).

        Returns:
            Column: The equivalent root-section column.
        """
        return Column.circular(self.minor_diameter, self.length,
                               end_condition=end_condition,
                               material=self.material)

    def slenderness_ratio(self, end_condition=1.0):
        """
        Column slenderness ratio Le / k of the screw.

        Uses the radius of gyration of the root section k = dr/4 and the
        effective length ``Le = end_condition * length``.

        Args:
            end_condition (float): Effective-length factor K (1.0 for
                pinned-pinned, 0.5 fixed-fixed, 2.0 fixed-free).

        Returns:
            float: Slenderness ratio (dimensionless).
        """
        return self.buckling_column(end_condition).slenderness_ratio

    def johnson_transition_ratio(self):
        """
        Slenderness ratio at the Euler/Johnson transition.

        ``(Le/k)_transition = sqrt(2*pi^2*E / Sy)``. Above this the Euler
        formula applies; below it use the J.B. Johnson formula for
        intermediate columns.

        Returns:
            float: Transition slenderness ratio (dimensionless).
        """
        return self.buckling_column().transition_slenderness

    def critical_buckling_load(self, end_condition=1.0):
        """
        Euler critical buckling load of the screw column.

        ``P_cr = pi^2 * E * I / Le^2`` with ``I = pi*dr^4/64`` and
        ``Le = end_condition * length``. Valid for slender columns; check
        :meth:`slenderness_ratio` against :meth:`johnson_transition_ratio`
        and use the Johnson formula for intermediate slenderness.

        Args:
            end_condition (float): Effective-length factor K (default 1.0).

        Returns:
            float: Critical load in N.
        """
        return self.buckling_column(end_condition).euler_load

    def check_buckling(self, force, end_condition=1.0):
        """
        Safety factor against Euler buckling for a compressive load.

        Args:
            force (float): Compressive axial load in N.
            end_condition (float): Effective-length factor K (default 1.0).

        Returns:
            float: Ratio of critical load to the applied load.

        Raises:
            ValueError: If ``force`` is zero.
        """
        if force == 0:
            raise ValueError("Force must be non-zero to compute a buckling factor")
        return self.buckling_column(end_condition).euler_safety_factor(abs(force))

    def screw_safety_factor(self, force, torque):
        """
        Safety factor against yielding of the screw body (von Mises).

        Uses the material yield strength (converted Pa -> MPa), unlike the
        inherited :meth:`~mecapy.base.MechaElement.safety_factor` which
        expects a stress in Pa.

        Args:
            force (float): Axial load F in N.
            torque (float): Torque carried by the screw body in N*mm.

        Returns:
            float: Ratio of yield strength to the equivalent stress.

        Raises:
            ValueError: If the equivalent stress is zero.
        """
        sigma_eq = self.von_mises_stress(force, torque)
        if sigma_eq == 0:
            raise ValueError("Equivalent stress is zero; safety factor is undefined")
        sy = self.material_properties["yield_strength"] / 1e6
        return sy / sigma_eq

    # ---- Internal helpers ----

    @staticmethod
    def _check_friction(mu, mu_collar, collar_diameter):
        """Validate friction coefficients (mu, and mu_collar if a collar is used)."""
        if mu < 0:
            raise ValueError("Friction coefficient must be non-negative")
        if collar_diameter is not None and mu_collar < 0:
            raise ValueError("Collar friction coefficient must be non-negative")

    @staticmethod
    def _collar_torque(force, collar_diameter, mu_collar):
        """Return the collar friction torque F*mu_c*dc/2 (0 if no collar)."""
        if collar_diameter is None:
            return 0.0
        return force * mu_collar * collar_diameter / 2

    def _side_diameter(self, side):
        """Return the diameter used for thread stresses on the given side."""
        if side == "screw":
            return self.minor_diameter
        if side == "nut":
            return self.major_diameter
        raise ValueError("side must be 'screw' or 'nut'")

    def __repr__(self):
        return (
            f"PowerScrew(major_diameter={self.major_diameter}mm, pitch={self.pitch}mm, "
            f"n_starts={self.n_starts}, thread_form={self.thread_form!r}, "
            f"material={self.material!r})"
        )
