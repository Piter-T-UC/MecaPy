"""Straight bevel gears.

Bevel gears transmit power between intersecting shafts (usually at 90
degrees). Tooth size tapers along the face, so the module and pitch
diameter are referred to the *large end* of the tooth.

Scope: geometry, kinematics and a simplified Lewis bending check using
Tredgold's virtual-teeth approximation. This is an educational check,
not an AGMA 2003 bevel rating.
"""

import math

from .gear import Gear


class BevelGear(Gear):
    """
    Straight bevel gear (module at the large end).

    Cone geometry depends on the mating gear and shaft angle, so the
    cone-related quantities are pair methods (``*_with``) rather than
    properties.

    Attributes:
        teeth (int): Number of teeth.
        module (float): Module at the large end in mm.
        pressure_angle (float): Pressure angle in degrees.
        face_width (float): Face width along the cone in mm (None if
            not set).
        material (str): Material type.
    """

    _min_teeth = 4

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 face_width=None, material="steel", name=None,
                 diametral_pitch=None):
        """
        Initialize a straight bevel gear.

        Args:
            teeth (int): Number of teeth (>= 4).
            module (float): Module at the large end in mm (or give
                ``diametral_pitch``).
            pressure_angle (float): Pressure angle in degrees (default 20).
            face_width (float): Face width along the cone in mm.
            material (str): Material type (default: "steel").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to module.

        Raises:
            ValueError: For non-physical inputs.
        """
        super().__init__(teeth, module=module, pressure_angle=pressure_angle,
                         material=material, name=name,
                         diametral_pitch=diametral_pitch)
        if face_width is not None and face_width <= 0:
            raise ValueError("Face width must be strictly positive")
        self.face_width = face_width

    # ------------------------------------------------------------------
    # Pair (cone) geometry
    # ------------------------------------------------------------------

    def pitch_cone_angle_with(self, mate, shaft_angle=90.0):
        """
        Pitch cone angle of this gear meshing with a mate.

        ``gamma = atan(sin(sigma) / (N_mate/N_self + cos(sigma)))``,
        which reduces to ``atan(N_self / N_mate)`` for a 90-degree
        shaft angle.

        Args:
            mate (BevelGear): Mating bevel gear.
            shaft_angle (float): Angle between shafts in degrees
                (default: 90).

        Returns:
            float: Pitch cone angle in degrees.

        Raises:
            ValueError: If the gears cannot mesh or the shaft angle is
                out of (0, 180).
        """
        from .transmission import _check_mesh

        _check_mesh(self, mate)
        if not 0 < shaft_angle < 180:
            raise ValueError("Shaft angle must be between 0 and 180 degrees")
        sigma = math.radians(shaft_angle)
        return math.degrees(math.atan2(
            math.sin(sigma), mate.teeth / self.teeth + math.cos(sigma)
        ))

    def cone_distance_with(self, mate, shaft_angle=90.0):
        """
        Outer cone distance A0 (apex to the large end of the tooth).

        Args:
            mate (BevelGear): Mating bevel gear.
            shaft_angle (float): Angle between shafts in degrees.

        Returns:
            float: Cone distance in mm (d / (2 sin(gamma))).
        """
        gamma = math.radians(self.pitch_cone_angle_with(mate, shaft_angle))
        return self.pitch_diameter / (2 * math.sin(gamma))

    def mean_radius_with(self, mate, shaft_angle=90.0):
        """
        Pitch radius at the mid-face section.

        Args:
            mate (BevelGear): Mating bevel gear.
            shaft_angle (float): Angle between shafts in degrees.

        Returns:
            float: Mean pitch radius in mm.

        Raises:
            ValueError: If ``face_width`` is not set.
        """
        if self.face_width is None:
            raise ValueError("Face width must be set for the mean radius")
        gamma = math.radians(self.pitch_cone_angle_with(mate, shaft_angle))
        return (self.pitch_diameter / 2
                - (self.face_width / 2) * math.sin(gamma))

    def virtual_teeth_with(self, mate, shaft_angle=90.0):
        """
        Virtual (equivalent spur) tooth count, Tredgold approximation.

        Args:
            mate (BevelGear): Mating bevel gear.
            shaft_angle (float): Angle between shafts in degrees.

        Returns:
            float: Virtual teeth ``Zv = Z / cos(gamma)``.
        """
        gamma = math.radians(self.pitch_cone_angle_with(mate, shaft_angle))
        return self.teeth / math.cos(gamma)

    # ------------------------------------------------------------------
    # Simplified strength check
    # ------------------------------------------------------------------

    def lewis_bending_stress(self, tangential_force, mate, shaft_angle=90.0):
        """
        Simplified Lewis bending stress at the mid-face section.

        Evaluates the plain Lewis equation ``sigma = Ft / (b * m * Y)``
        with the module taken at the mean section
        (``m_mean = m * (1 - b / (2 A0))``) and the form factor Y looked
        up at the Tredgold virtual tooth count. Educational
        approximation — not an AGMA 2003 bevel rating.

        Args:
            tangential_force (float): Tangential force at the mean
                radius in N.
            mate (BevelGear): Mating bevel gear.
            shaft_angle (float): Angle between shafts in degrees.

        Returns:
            float: Bending stress in MPa.

        Raises:
            ValueError: If ``face_width`` is not set, or exceeds A0/3
                (standard bevel practice).
        """
        from .agma_data import lewis_form_factor

        if self.face_width is None:
            raise ValueError("Face width must be set for the bending check")
        a0 = self.cone_distance_with(mate, shaft_angle)
        if self.face_width > a0 / 3:
            raise ValueError(
                f"Face width {self.face_width} mm exceeds A0/3 = {a0 / 3:.2f} mm"
            )
        m_mean = self.module * (1 - self.face_width / (2 * a0))
        y = lewis_form_factor(self.virtual_teeth_with(mate, shaft_angle))
        return tangential_force / (self.face_width * m_mean * y)

    def force_report(self, power_kw, speed_rpm, mate, shaft_angle=90.0):
        """
        Forces and moments this bevel gear generates, at the mean radius.

        The tangential force acts at the mid-face (mean) radius; the
        radial and axial components follow the standard straight-bevel
        resolution (Shigley), which splits ``Ft * tan(alpha)`` between
        the radial and axial directions by the pitch cone angle:
        ``Fr = Ft tan(alpha) cos(gamma)``, ``Fa = Ft tan(alpha)
        sin(gamma)``. Because the geometry depends on the mate and shaft
        angle, they are required (like the other bevel ``*_with``
        methods).

        Args:
            power_kw (float): Transmitted power in kW.
            speed_rpm (float): Rotational speed of this gear in rpm.
            mate (BevelGear): Mating bevel gear.
            shaft_angle (float): Angle between shafts in degrees
                (default 90).

        Returns:
            dict: With keys ``Ft``, ``Fr``, ``Fa`` (N), ``torque`` and
            ``moment`` (N*m) and ``pitch_line_velocity`` (m/s, at the
            mean radius).

        Raises:
            ValueError: If power or speed is not strictly positive, the
                gears cannot mesh, or ``face_width`` is not set.
        """
        if power_kw <= 0:
            raise ValueError("Power must be strictly positive")
        if speed_rpm <= 0:
            raise ValueError("Speed must be strictly positive")
        r_mean = self.mean_radius_with(mate, shaft_angle)  # mm
        gamma = math.radians(self.pitch_cone_angle_with(mate, shaft_angle))
        phi = math.radians(self.pressure_angle)
        v = 2 * math.pi * speed_rpm / 60.0 * r_mean / 1000.0  # m/s
        ft = 1000 * power_kw / v
        fr = ft * math.tan(phi) * math.cos(gamma)
        fa = ft * math.tan(phi) * math.sin(gamma)
        return {
            "Ft": ft,
            "Fr": fr,
            "Fa": fa,
            "torque": ft * r_mean / 1000.0,
            "moment": fa * r_mean / 1000.0,
            "pitch_line_velocity": v,
        }

    def describe_forces(self, power_kw, speed_rpm, mate, shaft_angle=90.0):
        """
        Multi-line report of the forces and moments (bevel, mean radius).

        Formatted like :meth:`Gear.describe_forces` but evaluated at the
        mean radius for the given mate and shaft angle.

        Args:
            power_kw (float): Transmitted power in kW.
            speed_rpm (float): Rotational speed of this gear in rpm.
            mate (BevelGear): Mating bevel gear.
            shaft_angle (float): Angle between shafts in degrees.

        Returns:
            str: Formatted force/moment report.
        """
        f = self.force_report(power_kw, speed_rpm, mate, shaft_angle)
        header = f"{self.__class__.__name__} forces and moments"
        if self.name:
            header += f" '{self.name}'"
        return "\n".join([
            header,
            "=" * 40,
            f"transmitted power (P) = {power_kw:.3f} kW",
            f"rotational speed (n) = {speed_rpm:.1f} rpm",
            f"pitch-line velocity at mean radius (v) = "
            f"{f['pitch_line_velocity']:.3f} m/s",
            f"tangential force (Ft) = {f['Ft']:.1f} N",
            f"radial force (Fr) = {f['Fr']:.1f} N",
            f"axial force (Fa) = {f['Fa']:.1f} N",
            f"torque (T) = {f['torque']:.3f} N*m",
            f"axial moment (Ma) = {f['moment']:.3f} N*m",
        ])

    def __repr__(self):
        return (
            f"BevelGear(teeth={self.teeth}, module={self.module}, "
            f"material={self.material!r})"
        )
