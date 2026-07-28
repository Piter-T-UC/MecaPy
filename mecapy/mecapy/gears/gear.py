"""Gear design and analysis module.

Defines the :class:`Gear` base class shared by all gear types
(spur, helical, herringbone, bevel, worm wheel, ...). Geometry follows
the standard full-depth tooth system.

Units: SI-metric is primary (module in mm, angles in degrees). US
customary users can pass ``diametral_pitch`` (teeth/inch) instead of
``module``; it is converted internally so all downstream code is metric.
See :mod:`mecapy.utils.converters` for force/stress/power conversions.
"""

import math

from ..base import MechaElement


def involute(angle):
    """
    Involute function inv(angle) = tan(angle) - angle.

    Args:
        angle (float): Angle in radians.

    Returns:
        float: Involute of the angle (dimensionless).
    """
    return math.tan(angle) - angle


def inverse_involute(value):
    """
    Angle whose involute equals ``value`` (inverse of :func:`involute`).

    Solved by Newton iteration on ``tan(x) - x - value = 0``.

    Args:
        value (float): Involute value (dimensionless, >= 0).

    Returns:
        float: Angle in radians.

    Raises:
        ValueError: If ``value`` is negative.
    """
    if value < 0:
        raise ValueError("Involute value must be non-negative")
    if value == 0:
        return 0.0
    angle = 0.3
    for _ in range(100):
        f = math.tan(angle) - angle - value
        if abs(f) < 1e-12:
            break
        angle -= f / math.tan(angle) ** 2
    return angle


class Gear(MechaElement):
    """
    Base class for gears (standard full-depth involute teeth).

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`. Direct use is equivalent to a
    generic cylindrical gear; prefer the specific subclasses
    (:class:`~mecapy.gears.SpurGear`, :class:`~mecapy.gears.HelicalGear`,
    ...) for real work.

    Attributes:
        teeth (int): Number of teeth.
        module (float): Gear module in mm.
        pressure_angle (float): Pressure angle in degrees (default 20).
        profile_shift (float): Profile shift coefficient x (dimensionless,
            0 for standard teeth). Settable through the cylindrical gear
            constructors; always 0 for other gear types.
        internal (bool): True for an internal (ring) gear, whose teeth
            point inwards. Settable through the cylindrical gear
            constructors; always False for other gear types.
        material (str): Material type.
    """

    #: Minimum number of teeth accepted by the constructor. Subclasses
    #: override (e.g. cylindrical gears require more; worms allow 1 start).
    _min_teeth = 1

    #: External gear by default. :class:`~mecapy.gears.CylindricalGear`
    #: sets an instance attribute that shadows this.
    internal = False

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 material="steel", name=None, diametral_pitch=None,
                 power_kw=None, speed_rpm=None):
        """
        Initialize a Gear object.

        Args:
            teeth (int): Number of teeth. Must be >= 1.
            module (float): Gear module in mm. Give exactly one of
                ``module`` or ``diametral_pitch``.
            pressure_angle (float): Pressure angle in degrees
                (default: 20). Must satisfy 0 < angle < 45.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the gear.
            diametral_pitch (float): US-customary alternative to
                ``module``, in teeth per inch. Stored internally as
                ``module = 25.4 / diametral_pitch``.
            power_kw (float): Optional transmitted power in kW. When set,
                the force methods use it as the default (no need to pass
                it at call time). A :class:`Transmission` fills this in
                for every downstream gear from the first gear.
            speed_rpm (float): Optional rotational speed in rpm, used the
                same way as ``power_kw``.

        Raises:
            ValueError: If teeth/module/pressure angle are non-physical,
                if both (or neither) of ``module`` and ``diametral_pitch``
                are given, or if ``power_kw``/``speed_rpm`` are given but
                not strictly positive.
        """
        super().__init__(name=name, material=material)
        if teeth != int(teeth) or teeth < self._min_teeth:
            raise ValueError(
                f"Teeth must be an integer >= {self._min_teeth}, got {teeth}"
            )
        if (module is None) == (diametral_pitch is None):
            raise ValueError(
                "Give exactly one of 'module' (mm) or 'diametral_pitch' (1/in)"
            )
        if diametral_pitch is not None:
            if diametral_pitch <= 0:
                raise ValueError("Diametral pitch must be strictly positive")
            module = 25.4 / diametral_pitch
        if module <= 0:
            raise ValueError("Module must be strictly positive")
        if not 0 < pressure_angle < 45:
            raise ValueError("Pressure angle must be between 0 and 45 degrees")
        self.teeth = int(teeth)
        self.module = module
        self.pressure_angle = pressure_angle
        self.profile_shift = 0.0
        self.power_kw = power_kw
        self.speed_rpm = speed_rpm

    @property
    def power_kw(self):
        """float or None: Transmitted power in kW (None if unset)."""
        return self._power_kw

    @power_kw.setter
    def power_kw(self, value):
        if value is not None and value <= 0:
            raise ValueError("Power must be strictly positive")
        self._power_kw = value

    @property
    def speed_rpm(self):
        """float or None: Rotational speed in rpm (None if unset)."""
        return self._speed_rpm

    @speed_rpm.setter
    def speed_rpm(self, value):
        if value is not None and value <= 0:
            raise ValueError("Speed (rpm) must be strictly positive")
        self._speed_rpm = value

    def change_teeth(self, teeth):
        """
        Set a new tooth count on the same gear (for design iteration).

        Every derived quantity is a computed property, so pitch, base,
        outside and root diameters, undercut checks, pair-wise mesh
        methods and :meth:`describe` all reflect the new count
        immediately — just recompute after calling this.

        Args:
            teeth (int): New number of teeth. Same validity rule as the
                constructor (integer >= the type's minimum).

        Returns:
            Gear: ``self``, so calls can be chained, e.g.
            ``gear.change_teeth(20).pitch_diameter``.

        Raises:
            ValueError: If ``teeth`` is not a valid tooth count.
        """
        if teeth != int(teeth) or teeth < self._min_teeth:
            raise ValueError(
                f"Teeth must be an integer >= {self._min_teeth}, got {teeth}"
            )
        self.teeth = int(teeth)
        return self

    # ------------------------------------------------------------------
    # Geometry (standard full-depth system, dimensions in mm)
    # ------------------------------------------------------------------

    @property
    def diametral_pitch(self):
        """float: Diametral pitch in teeth per inch (25.4 / module)."""
        return 25.4 / self.module

    @property
    def pitch_diameter(self):
        """float: Pitch diameter in mm (teeth * module)."""
        return self.teeth * self.module

    @property
    def circular_pitch(self):
        """float: Circular pitch in mm (pi * module)."""
        return math.pi * self.module

    @property
    def addendum(self):
        """float: Addendum in mm, ha = m * (1 + x)."""
        return self.module * (1 + self.profile_shift)

    @property
    def dedendum(self):
        """float: Dedendum in mm, hf = m * (1.25 - x)."""
        return self.module * (1.25 - self.profile_shift)

    @property
    def _sign(self):
        """int: +1 for an external gear, -1 for an internal (ring) gear.

        The addendum and dedendum keep their magnitudes on an internal
        gear; only the direction in which they are applied flips.
        """
        return -1 if self.internal else 1

    @property
    def outside_diameter(self):
        """float: Outside (tip) diameter in mm.

        ``d + 2 ha`` for an external gear. On an internal gear the tip
        circle lies *inside* the pitch circle, so it is ``d - 2 ha``.
        """
        return self.pitch_diameter + self._sign * 2 * self.addendum

    @property
    def root_diameter(self):
        """float: Root diameter in mm.

        ``d - 2 hf`` for an external gear. On an internal gear the root
        circle lies *outside* the pitch circle, so it is ``d + 2 hf``
        and ``outside_diameter < pitch_diameter < root_diameter``.
        """
        return self.pitch_diameter - self._sign * 2 * self.dedendum

    @property
    def whole_depth(self):
        """float: Whole tooth depth in mm (addendum + dedendum, 2.25 * m)."""
        return self.addendum + self.dedendum

    @property
    def base_diameter(self):
        """float: Base circle diameter in mm (d * cos(pressure angle))."""
        return self.pitch_diameter * math.cos(math.radians(self.pressure_angle))

    @property
    def clearance(self):
        """float: Bottom clearance in mm (c = 0.25 * m, independent of x)."""
        return 0.25 * self.module

    @property
    def working_depth(self):
        """float: Working depth in mm (hk = 2 * m, basic rack)."""
        return 2.0 * self.module

    @property
    def tooth_thickness(self):
        """float: Arc tooth thickness at the pitch circle in mm.

        s = m * (pi / 2 + 2 * x * tan(alpha)). For helical gears this is
        the normal tooth thickness. On an internal gear tooth and space
        swap roles, so the shift term changes sign:
        s = m * (pi / 2 - 2 * x * tan(alpha)).
        """
        alpha = math.radians(self.pressure_angle)
        return self.module * (
            math.pi / 2
            + self._sign * 2 * self.profile_shift * math.tan(alpha)
        )

    @property
    def base_pitch(self):
        """float: Base pitch in mm (pb = pi * m * cos(alpha)).

        Normal base pitch for helical gears.
        """
        return math.pi * self.module * math.cos(
            math.radians(self.pressure_angle))

    @property
    def pitch_radius(self):
        """float: Pitch radius in mm (pitch_diameter / 2)."""
        return self.pitch_diameter / 2

    @property
    def base_radius(self):
        """float: Base circle radius in mm (base_diameter / 2)."""
        return self.base_diameter / 2

    @property
    def outside_radius(self):
        """float: Outside (tip) radius in mm (outside_diameter / 2)."""
        return self.outside_diameter / 2

    @property
    def root_radius(self):
        """float: Root radius in mm (root_diameter / 2)."""
        return self.root_diameter / 2

    # ------------------------------------------------------------------
    # Forces and moments
    # ------------------------------------------------------------------

    def _resolve_speed(self, speed_rpm):
        """Fall back to the stored ``speed_rpm`` when none is passed."""
        speed_rpm = self.speed_rpm if speed_rpm is None else speed_rpm
        if speed_rpm is None:
            raise ValueError(
                "No speed_rpm set on this gear; pass it, set gear.speed_rpm, "
                "or add the gear to a Transmission"
            )
        return speed_rpm

    def _resolve_power(self, power_kw):
        """Fall back to the stored ``power_kw`` when none is passed."""
        power_kw = self.power_kw if power_kw is None else power_kw
        if power_kw is None:
            raise ValueError(
                "No power_kw set on this gear; pass it, set gear.power_kw, "
                "or add the gear to a Transmission"
            )
        return power_kw

    def pitch_line_velocity(self, speed_rpm=None):
        """
        Pitch-line velocity at a given rotational speed.

        Args:
            speed_rpm (float): Rotational speed in rpm. Defaults to the
                gear's stored ``speed_rpm`` when omitted.

        Returns:
            float: Pitch-line velocity in m/s.

        Raises:
            ValueError: If no speed is passed and none is stored.
        """
        speed_rpm = self._resolve_speed(speed_rpm)
        return math.pi * self.pitch_diameter * speed_rpm / 60000.0

    def tangential_force(self, power_kw=None, speed_rpm=None):
        """
        Tangential force at the pitch circle for a transmitted power.

        Args:
            power_kw (float): Transmitted power in kW. Defaults to the
                gear's stored ``power_kw`` when omitted.
            speed_rpm (float): Rotational speed of this gear in rpm.
                Defaults to the gear's stored ``speed_rpm`` when omitted.

        Returns:
            float: Tangential force Ft in N.

        Raises:
            ValueError: If power or speed is not strictly positive, or
                is neither passed nor stored on the gear.
        """
        power_kw = self._resolve_power(power_kw)
        speed_rpm = self._resolve_speed(speed_rpm)
        if power_kw <= 0:
            raise ValueError("Power must be strictly positive")
        if speed_rpm <= 0:
            raise ValueError("Speed must be strictly positive")
        v = self.pitch_line_velocity(speed_rpm)
        return 1000 * power_kw / v

    def radial_force(self, tangential_force):
        """float: Radial (separating) force in N: Ft * tan(alpha)."""
        return tangential_force * math.tan(math.radians(self.pressure_angle))

    def axial_force(self, tangential_force):
        """float: Axial (thrust) force in N. Zero for a plain gear (no
        helix)."""
        return 0.0

    def force_report(self, power_kw=None, speed_rpm=None):
        """
        Forces and moments this gear generates at a working point.

        Collects the tangential, radial (separating) and axial (thrust)
        tooth forces together with the transmitted torque and the
        overturning moment produced by the thrust. Reuses
        :meth:`tangential_force`, :meth:`radial_force` and
        :meth:`axial_force`, so subclasses that override those
        (helical thrust, herringbone cancellation, ...) are reflected
        automatically.

        Args:
            power_kw (float): Transmitted power in kW. Defaults to the
                gear's stored ``power_kw`` when omitted.
            speed_rpm (float): Rotational speed of this gear in rpm.
                Defaults to the gear's stored ``speed_rpm`` when omitted.

        Returns:
            dict: With keys ``Ft``, ``Fr``, ``Fa`` (forces in N),
            ``torque`` and ``moment`` (in N*m),
            ``pitch_line_velocity`` (in m/s) and ``internal`` (bool).
            ``torque`` is Ft times the pitch radius; ``moment`` is the
            thrust Fa times the pitch radius (0 without an axial force).
            All forces are magnitudes; ``internal`` flags that the
            radial force is directed *towards* the gear centre (the mesh
            pulls the ring inwards) instead of away from it.

        Raises:
            ValueError: If power or speed is not strictly positive.
        """
        ft = self.tangential_force(power_kw, speed_rpm)
        fr = self.radial_force(ft)
        fa = self.axial_force(ft)
        r = self.pitch_radius  # mm
        return {
            "Ft": ft,
            "Fr": fr,
            "Fa": fa,
            "torque": ft * r / 1000.0,
            "moment": fa * r / 1000.0,
            "pitch_line_velocity": self.pitch_line_velocity(speed_rpm),
            "internal": bool(self.internal),
        }

    def describe_forces(self, power_kw=None, speed_rpm=None):
        """
        Multi-line report of the forces and moments this gear generates.

        Formatted like :meth:`describe` (``label (symbol) = value unit``
        lines) from :meth:`force_report`. The string is returned, not
        printed.

        Args:
            power_kw (float): Transmitted power in kW. Defaults to the
                gear's stored ``power_kw`` when omitted.
            speed_rpm (float): Rotational speed of this gear in rpm.
                Defaults to the gear's stored ``speed_rpm`` when omitted.

        Returns:
            str: Formatted force/moment report.
        """
        f = self.force_report(power_kw, speed_rpm)
        power_kw = self._resolve_power(power_kw)
        speed_rpm = self._resolve_speed(speed_rpm)
        header = f"{self.__class__.__name__} forces and moments"
        if self.name:
            header += f" '{self.name}'"
        lines = [
            header,
            "=" * 40,
            f"transmitted power (P) = {power_kw:.3f} kW",
            f"rotational speed (n) = {speed_rpm:.1f} rpm",
            f"pitch-line velocity (v) = {f['pitch_line_velocity']:.3f} m/s",
            f"tangential force (Ft) = {f['Ft']:.1f} N",
            f"radial force (Fr) = {f['Fr']:.1f} N",
            f"axial force (Fa) = {f['Fa']:.1f} N",
            f"torque (T) = {f['torque']:.3f} N*m",
            f"axial moment (Ma) = {f['moment']:.3f} N*m",
        ]
        if f["internal"]:
            lines.append(
                "note: internal mesh - the radial force is directed "
                "towards the gear centre"
            )
        return "\n".join(lines)

    def describe(self):
        """
        Multi-line summary of every geometry parameter.

        Each line is formatted as ``label (symbol) = value unit``, e.g.
        ``addendum (ha) = 2.500 mm``. The string is returned, not
        printed; use ``print(gear.describe())``.

        Returns:
            str: Formatted geometry report.
        """
        header = f"{self.__class__.__name__} geometry"
        if self.name:
            header += f" '{self.name}'"
        lines = [
            header,
            "=" * 40,
            f"number of teeth (z) = {self.teeth}",
            f"module (m) = {self.module:.3f} mm",
            f"pressure angle (alpha) = {self.pressure_angle:.3f} deg",
            f"profile shift coefficient (x) = {self.profile_shift:.3f}",
            f"pitch diameter (d) = {self.pitch_diameter:.3f} mm",
            f"base diameter (db) = {self.base_diameter:.3f} mm",
            f"outside diameter (da) = {self.outside_diameter:.3f} mm",
            f"root diameter (df) = {self.root_diameter:.3f} mm",
            f"circular pitch (p) = {self.circular_pitch:.3f} mm",
            f"base pitch (pb) = {self.base_pitch:.3f} mm",
            f"tooth thickness (s) = {self.tooth_thickness:.3f} mm",
            f"addendum (ha) = {self.addendum:.3f} mm",
            f"dedendum (hf) = {self.dedendum:.3f} mm",
            f"whole depth (h) = {self.whole_depth:.3f} mm",
            f"working depth (hk) = {self.working_depth:.3f} mm",
            f"clearance (c) = {self.clearance:.3f} mm",
        ]
        return "\n".join(lines)

    def __repr__(self):
        shift = (f", x={self.profile_shift}"
                 if self.profile_shift != 0 else "")
        return (
            f"{self.__class__.__name__}(teeth={self.teeth}, "
            f"module={self.module}{shift}, material={self.material!r})"
        )
