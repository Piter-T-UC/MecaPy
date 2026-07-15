"""Cylindrical (parallel-axis) gears: spur, helical and herringbone.

For helical gears the constructor ``module`` and ``pressure_angle`` are
the *normal* module and *normal* pressure angle; transverse quantities
are exposed as properties. Spur gears are the ``helix_angle = 0`` case.

Units: mm, degrees, N, MPa, rpm (see :mod:`mecapy.gears.gear`).
"""

import math

from .gear import Gear


class CylindricalGear(Gear):
    """
    Involute cylindrical gear (parallel-axis family).

    Shared machinery for :class:`SpurGear`, :class:`HelicalGear` and
    :class:`HerringboneGear` — transverse geometry, contact ratio,
    force analysis and AGMA rating. Usually instantiated through those
    subclasses.

    Attributes:
        teeth (int): Number of teeth.
        module (float): Normal module in mm.
        pressure_angle (float): Normal pressure angle in degrees.
        helix_angle (float): Helix angle in degrees (0 for spur).
        hand (str): Helix hand, "right" or "left" (None for spur and
            herringbone).
        face_width (float): Face width in mm (None if not set).
        material (str): Material type.
    """

    _min_teeth = 4

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 helix_angle=0.0, hand=None, face_width=None,
                 material="steel", name=None, diametral_pitch=None):
        """
        Initialize a cylindrical gear.

        Args:
            teeth (int): Number of teeth (>= 4).
            module (float): Normal module in mm. Give exactly one of
                ``module`` or ``diametral_pitch``.
            pressure_angle (float): Normal pressure angle in degrees
                (default: 20).
            helix_angle (float): Helix angle in degrees, 0 <= angle <= 45
                (default: 0, i.e. spur).
            hand (str): Helix hand, "right" or "left". Required when
                ``helix_angle > 0`` (except herringbone), must be None
                for spur gears.
            face_width (float): Face width in mm. Required for AGMA
                rating, optional for pure geometry.
            material (str): Material type (default: "steel").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to
                ``module`` (normal diametral pitch, teeth/inch).

        Raises:
            ValueError: For non-physical inputs or inconsistent
                helix_angle/hand combinations.
        """
        super().__init__(teeth, module=module, pressure_angle=pressure_angle,
                         material=material, name=name,
                         diametral_pitch=diametral_pitch)
        if not 0 <= helix_angle <= 45:
            raise ValueError("Helix angle must be between 0 and 45 degrees")
        if hand is not None and hand not in ("right", "left"):
            raise ValueError("Hand must be 'right', 'left' or None")
        if helix_angle == 0 and hand is not None:
            raise ValueError("A spur gear (helix_angle=0) has no hand")
        if face_width is not None and face_width <= 0:
            raise ValueError("Face width must be strictly positive")
        self.helix_angle = helix_angle
        self.hand = hand
        self.face_width = face_width

    # ------------------------------------------------------------------
    # Transverse geometry
    # ------------------------------------------------------------------

    @property
    def transverse_module(self):
        """float: Transverse module in mm (mn / cos(helix angle))."""
        return self.module / math.cos(math.radians(self.helix_angle))

    @property
    def transverse_pressure_angle(self):
        """float: Transverse pressure angle in degrees.

        From ``tan(phi_t) = tan(phi_n) / cos(beta)``.
        """
        phi_n = math.radians(self.pressure_angle)
        beta = math.radians(self.helix_angle)
        return math.degrees(math.atan(math.tan(phi_n) / math.cos(beta)))

    @property
    def pitch_diameter(self):
        """float: Pitch diameter in mm (teeth * transverse module)."""
        return self.teeth * self.transverse_module

    @property
    def base_diameter(self):
        """float: Base circle diameter in mm (d * cos(phi_t))."""
        phi_t = math.radians(self.transverse_pressure_angle)
        return self.pitch_diameter * math.cos(phi_t)

    @property
    def axial_pitch(self):
        """float: Axial pitch in mm (helical only).

        Raises:
            ValueError: For a spur gear (no axial advance).
        """
        if self.helix_angle == 0:
            raise ValueError("A spur gear has no axial pitch")
        beta = math.radians(self.helix_angle)
        return math.pi * self.module / math.sin(beta)

    @property
    def lead(self):
        """float: Lead of the helix in mm (helical only)."""
        return self.axial_pitch * self.teeth

    @property
    def min_teeth_no_undercut(self):
        """int: Minimum pinion teeth to avoid undercutting.

        Uses the interference limit ``2 / sin^2(phi)`` in the transverse
        plane, reduced by ``cos(beta)`` for helical gears, rounded up
        (ceil). At 20 degrees this gives 18; the commonly quoted
        practical value is 17.
        """
        phi_t = math.radians(self.transverse_pressure_angle)
        beta = math.radians(self.helix_angle)
        return math.ceil(2 * math.cos(beta) / math.sin(phi_t) ** 2)

    # ------------------------------------------------------------------
    # Pair (mesh) calculations
    # ------------------------------------------------------------------

    def center_distance_with(self, other):
        """
        Center distance for an external mesh with another gear.

        Args:
            other (CylindricalGear): Mating gear.

        Returns:
            float: Center distance in mm ((d1 + d2) / 2).

        Raises:
            ValueError: If the two gears cannot mesh.
        """
        from .transmission import _check_mesh

        _check_mesh(self, other)
        return (self.pitch_diameter + other.pitch_diameter) / 2

    def contact_ratio_with(self, other):
        """
        Transverse contact ratio for an external mesh with another gear.

        Length of the line of action divided by the transverse base
        pitch. For helical gears use also :meth:`axial_contact_ratio_with`
        and :meth:`total_contact_ratio_with`.

        Args:
            other (CylindricalGear): Mating gear.

        Returns:
            float: Transverse contact ratio (dimensionless, > 1 for a
            correctly proportioned mesh).

        Raises:
            ValueError: If the two gears cannot mesh.
        """
        from .transmission import _check_mesh

        _check_mesh(self, other)
        phi_t = math.radians(self.transverse_pressure_angle)
        c = (self.pitch_diameter + other.pitch_diameter) / 2
        line_of_action = (
            math.sqrt((self.outside_diameter / 2) ** 2
                      - (self.base_diameter / 2) ** 2)
            + math.sqrt((other.outside_diameter / 2) ** 2
                        - (other.base_diameter / 2) ** 2)
            - c * math.sin(phi_t)
        )
        base_pitch = math.pi * self.transverse_module * math.cos(phi_t)
        return line_of_action / base_pitch

    def axial_contact_ratio_with(self, other):
        """
        Axial (face) contact ratio for a helical mesh.

        Args:
            other (CylindricalGear): Mating gear (used for the mesh
                compatibility check).

        Returns:
            float: Axial contact ratio ``b * tan(beta) / (pi * mt)``.

        Raises:
            ValueError: If the gears cannot mesh, the gear is spur, or
                ``face_width`` is not set.
        """
        from .transmission import _check_mesh

        _check_mesh(self, other)
        if self.helix_angle == 0:
            raise ValueError("Axial contact ratio applies to helical gears only")
        if self.face_width is None:
            raise ValueError("Face width must be set for the axial contact ratio")
        beta = math.radians(self.helix_angle)
        return (self.face_width * math.tan(beta)
                / (math.pi * self.transverse_module))

    def total_contact_ratio_with(self, other):
        """float: Sum of transverse and axial contact ratios (helical)."""
        return (self.contact_ratio_with(other)
                + self.axial_contact_ratio_with(other))

    # ------------------------------------------------------------------
    # Forces and rating
    # ------------------------------------------------------------------

    def tangential_force(self, power_kw, speed_rpm):
        """
        Tangential force at the pitch circle for a transmitted power.

        Args:
            power_kw (float): Transmitted power in kW.
            speed_rpm (float): Rotational speed of this gear in rpm.

        Returns:
            float: Tangential force Ft in N.

        Raises:
            ValueError: If power or speed is not strictly positive.
        """
        if power_kw <= 0:
            raise ValueError("Power must be strictly positive")
        if speed_rpm <= 0:
            raise ValueError("Speed must be strictly positive")
        v = self.pitch_line_velocity(speed_rpm)
        return 1000 * power_kw / v

    def pitch_line_velocity(self, speed_rpm):
        """
        Pitch-line velocity at a given rotational speed.

        Args:
            speed_rpm (float): Rotational speed in rpm.

        Returns:
            float: Pitch-line velocity in m/s.
        """
        return math.pi * self.pitch_diameter * speed_rpm / 60000.0

    def radial_force(self, tangential_force):
        """float: Radial (separating) force in N: Ft * tan(phi_t)."""
        phi_t = math.radians(self.transverse_pressure_angle)
        return tangential_force * math.tan(phi_t)

    def axial_force(self, tangential_force):
        """float: Axial (thrust) force in N: Ft * tan(beta)."""
        return tangential_force * math.tan(math.radians(self.helix_angle))

    def rate_agma(self, mate, **kwargs):
        """
        AGMA bending and pitting rating of this gear (pinion) with a mate.

        Convenience wrapper around
        :class:`~mecapy.gears.agma.AGMARating`; ``self`` is taken as the
        pinion (driver).

        Args:
            mate (CylindricalGear or Rack): Mating gear.
            **kwargs: Forwarded to
                :class:`~mecapy.gears.agma.AGMARating` (power_kw,
                pinion_speed_rpm, Ko, Qv, ...).

        Returns:
            AGMARating: Fully evaluated rating object.
        """
        from .agma import AGMARating

        return AGMARating(self, mate, **kwargs)


class SpurGear(CylindricalGear):
    """
    Spur gear: straight teeth, parallel to the axis.

    A :class:`CylindricalGear` with ``helix_angle = 0`` and no hand.
    Transverse and normal quantities coincide.
    """

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 face_width=None, material="steel", name=None,
                 diametral_pitch=None):
        """
        Initialize a spur gear.

        Args:
            teeth (int): Number of teeth (>= 4).
            module (float): Module in mm (or give ``diametral_pitch``).
            pressure_angle (float): Pressure angle in degrees (default 20).
            face_width (float): Face width in mm (needed for AGMA rating).
            material (str): Material type (default: "steel").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to module.
        """
        super().__init__(teeth, module=module, pressure_angle=pressure_angle,
                         helix_angle=0.0, hand=None, face_width=face_width,
                         material=material, name=name,
                         diametral_pitch=diametral_pitch)


class HelicalGear(CylindricalGear):
    """
    Helical gear: teeth at a helix angle to the axis.

    ``module`` and ``pressure_angle`` are normal-plane values; two
    external helical gears mesh on parallel axes when they share module,
    pressure angle, helix angle and have *opposite* hands.
    """

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 helix_angle=None, hand=None, face_width=None,
                 material="steel", name=None, diametral_pitch=None):
        """
        Initialize a helical gear.

        Args:
            teeth (int): Number of teeth (>= 4).
            module (float): Normal module in mm (or ``diametral_pitch``).
            pressure_angle (float): Normal pressure angle in degrees.
            helix_angle (float): Helix angle in degrees, 0 < angle <= 45.
                Required.
            hand (str): Helix hand, "right" or "left". Required.
            face_width (float): Face width in mm.
            material (str): Material type (default: "steel").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to module.

        Raises:
            ValueError: If ``helix_angle`` or ``hand`` is missing or
                out of range.
        """
        if helix_angle is None or not 0 < helix_angle <= 45:
            raise ValueError(
                "A helical gear requires 0 < helix_angle <= 45 degrees"
            )
        if hand not in ("right", "left"):
            raise ValueError("A helical gear requires hand 'right' or 'left'")
        super().__init__(teeth, module=module, pressure_angle=pressure_angle,
                         helix_angle=helix_angle, hand=hand,
                         face_width=face_width, material=material, name=name,
                         diametral_pitch=diametral_pitch)


class HerringboneGear(HelicalGear):
    """
    Herringbone (double-helical) gear: two opposed helical halves.

    The opposed halves cancel axial thrust, so the gear has no hand and
    ``net_axial_force`` is zero. ``face_width`` is the total width of
    both halves. Meshes with another herringbone gear of equal helix
    angle.
    """

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 helix_angle=None, face_width=None, material="steel",
                 name=None, diametral_pitch=None):
        """
        Initialize a herringbone gear.

        Args:
            teeth (int): Number of teeth (>= 4).
            module (float): Normal module in mm (or ``diametral_pitch``).
            pressure_angle (float): Normal pressure angle in degrees.
            helix_angle (float): Helix angle of each half in degrees,
                0 < angle <= 45. Required.
            face_width (float): Total face width of both halves in mm.
            material (str): Material type (default: "steel").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to module.
        """
        super().__init__(teeth, module=module, pressure_angle=pressure_angle,
                         helix_angle=helix_angle, hand="right",
                         face_width=face_width, material=material, name=name,
                         diametral_pitch=diametral_pitch)
        # The opposed halves have no net hand.
        self.hand = None

    @property
    def net_axial_force(self):
        """float: Net axial force in N — always 0 (thrust cancels)."""
        return 0.0

    def axial_force(self, tangential_force):
        """float: Net axial force in N — always 0 for herringbone."""
        return 0.0
