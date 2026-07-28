"""Cylindrical (parallel-axis) gears: spur, helical and herringbone.

For helical gears the constructor ``module`` and ``pressure_angle`` are
the *normal* module and *normal* pressure angle; transverse quantities
are exposed as properties. Spur gears are the ``helix_angle = 0`` case.

Units: mm, degrees, N, MPa, rpm (see :mod:`mecapy.gears.gear`).
"""

import math

from .gear import Gear, inverse_involute, involute

#: Minimum tooth difference (z_ring - z_pinion) for a 20 degree
#: full-depth internal mesh, below which the pinion tip trims the ring
#: tooth tips during assembly. Classical shop rule (Buckingham); the
#: exact limit depends on the addendum system and profile shift.
_MIN_INTERNAL_TOOTH_DIFFERENCE = 12


def _mesh_terms(a, b):
    """
    Mesh sums generalized to internal meshes.

    For an external pair the classical mesh quantities are the sums
    ``z1 + z2``, ``x1 + x2`` and ``(d1 + d2) / 2``. When one member is
    an internal (ring) gear they become differences, ring minus pinion.
    Every pair method routes its sign convention through this helper.

    Args:
        a (CylindricalGear): One member of the mesh.
        b (CylindricalGear): The other member.

    Returns:
        tuple: ``(z_eff, x_eff, a_ref, internal)`` where ``internal`` is
        True when exactly one member is an internal gear, and ``a_ref``
        is the reference center distance in mm.
    """
    if a.internal or b.internal:
        ring, pinion = (a, b) if a.internal else (b, a)
        return (
            ring.teeth - pinion.teeth,
            ring.profile_shift - pinion.profile_shift,
            (ring.pitch_diameter - pinion.pitch_diameter) / 2,
            True,
        )
    return (
        a.teeth + b.teeth,
        a.profile_shift + b.profile_shift,
        (a.pitch_diameter + b.pitch_diameter) / 2,
        False,
    )


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
        profile_shift (float): Profile shift coefficient x
            (dimensionless, 0 for standard teeth).
        internal (bool): True for an internal (ring) gear.
        material (str): Material type.
    """

    _min_teeth = 4

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 helix_angle=0.0, hand=None, face_width=None,
                 profile_shift=0.0, internal=False, material="steel",
                 name=None, diametral_pitch=None, power_kw=None,
                 speed_rpm=None):
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
            profile_shift (float): Profile shift coefficient x,
                dimensionless, -1 < x < 1 (default 0 = standard teeth).
                Positive x thickens the tooth root and helps avoid
                undercut on small pinions.
            internal (bool): True for an internal (ring) gear, whose
                teeth point inwards (default: False). An internal gear
                meshes only with an external cylindrical gear of fewer
                teeth, and the mesh preserves the direction of rotation.
            material (str): Material type (default: "steel").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to
                ``module`` (normal diametral pitch, teeth/inch).

        Raises:
            ValueError: For non-physical inputs, inconsistent
                helix_angle/hand combinations, a profile shift outside
                (-1, 1), or an internal gear with so few teeth that its
                tip circle falls inside its base circle.
        """
        super().__init__(teeth, module=module, pressure_angle=pressure_angle,
                         material=material, name=name,
                         diametral_pitch=diametral_pitch,
                         power_kw=power_kw, speed_rpm=speed_rpm)
        if not 0 <= helix_angle <= 45:
            raise ValueError("Helix angle must be between 0 and 45 degrees")
        if hand is not None and hand not in ("right", "left"):
            raise ValueError("Hand must be 'right', 'left' or None")
        if helix_angle == 0 and hand is not None:
            raise ValueError("A spur gear (helix_angle=0) has no hand")
        if face_width is not None and face_width <= 0:
            raise ValueError("Face width must be strictly positive")
        if not -1.0 < profile_shift < 1.0:
            raise ValueError(
                "Profile shift coefficient must satisfy -1 < x < 1"
            )
        self.helix_angle = helix_angle
        self.hand = hand
        self.face_width = face_width
        self.profile_shift = profile_shift
        self.internal = bool(internal)
        if self.internal and self.outside_radius <= self.base_radius:
            raise ValueError(
                f"An internal gear with {self.teeth} teeth has its tip "
                f"circle inside its base circle; use more teeth or a "
                f"smaller profile shift"
            )

    def change_profile_shift(self, profile_shift):
        """
        Set a new profile shift coefficient x (for design iteration).

        Everything that depends on x is a computed property or a
        pair-wise method, so addendum, dedendum, outside/root diameter,
        tooth thickness, :attr:`is_undercut` and the working
        pressure-angle/center-distance pair methods all reflect the new
        value immediately.

        Args:
            profile_shift (float): New coefficient x, -1 < x < 1
                (0 = standard teeth), same rule as the constructor.

        Returns:
            CylindricalGear: ``self``, so calls can be chained, e.g.
            ``pinion.change_profile_shift(0.3).is_undercut``.

        Raises:
            ValueError: If ``profile_shift`` is outside (-1, 1), or if it
                would push an internal gear's tip circle inside its base
                circle (the value is left unchanged in that case).
        """
        if not -1.0 < profile_shift < 1.0:
            raise ValueError(
                "Profile shift coefficient must satisfy -1 < x < 1"
            )
        previous = self.profile_shift
        self.profile_shift = profile_shift
        if self.internal and self.outside_radius <= self.base_radius:
            self.profile_shift = previous
            raise ValueError(
                f"Profile shift x={profile_shift} puts this internal "
                f"gear's tip circle inside its base circle"
            )
        return self

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
    def transverse_base_pitch(self):
        """float: Transverse base pitch in mm (pi * mt * cos(phi_t))."""
        phi_t = math.radians(self.transverse_pressure_angle)
        return math.pi * self.transverse_module * math.cos(phi_t)

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
        practical value is 17. This is the standard-teeth (x = 0)
        guideline; :attr:`is_undercut` is the actual check that also
        accounts for profile shift.

        Raises:
            ValueError: On an internal gear, where undercut does not
                apply (see :meth:`has_trimming_interference_with`).
        """
        self._reject_if_internal("min_teeth_no_undercut")
        phi_t = math.radians(self.transverse_pressure_angle)
        beta = math.radians(self.helix_angle)
        return math.ceil(2 * math.cos(beta) / math.sin(phi_t) ** 2)

    @property
    def min_profile_shift(self):
        """float: Minimum profile shift coefficient to avoid undercut.

        ``x_min = 1 - z * sin^2(phi_t) / (2 * cos(beta))``; equivalently
        ``(Zmin - z) / Zmin`` with ``Zmin = 2 cos(beta) / sin^2(phi_t)``
        (the un-rounded value behind :attr:`min_teeth_no_undercut`).
        Negative when the gear already has enough teeth.

        Raises:
            ValueError: On an internal gear, where undercut does not
                apply (see :meth:`has_trimming_interference_with`).
        """
        self._reject_if_internal("min_profile_shift")
        phi_t = math.radians(self.transverse_pressure_angle)
        beta = math.radians(self.helix_angle)
        return 1 - self.teeth * math.sin(phi_t) ** 2 / (2 * math.cos(beta))

    @property
    def is_undercut(self):
        """bool: True when the tooth root is undercut (x < x_min).

        Always False for an internal gear: undercut is a property of
        generating an *external* tooth with a rack-type cutter. The
        equivalent internal failure mode is trimming interference, see
        :meth:`has_trimming_interference_with`.
        """
        if self.internal:
            return False
        return self.profile_shift < self.min_profile_shift - 1e-9

    def _reject_if_internal(self, what):
        """Raise for undercut-related quantities on an internal gear."""
        if self.internal:
            raise ValueError(
                f"'{what}' does not apply to internal gears (undercut is "
                f"an external-generation phenomenon); use "
                f"has_trimming_interference_with() instead"
            )

    # ------------------------------------------------------------------
    # Pair (mesh) calculations
    # ------------------------------------------------------------------

    def center_distance_with(self, other):
        """
        Reference center distance for a mesh with another gear.

        This is the unmodified distance ``(d1 + d2) / 2`` for an
        external pair and ``(d_ring - d_pinion) / 2`` when one member is
        an internal gear. For profile-shifted pairs the actual mounting
        distance is :meth:`working_center_distance_with`.

        Args:
            other (CylindricalGear): Mating gear.

        Returns:
            float: Reference center distance in mm.

        Raises:
            ValueError: If the two gears cannot mesh.
        """
        from .transmission import _check_mesh

        _check_mesh(self, other)
        return _mesh_terms(self, other)[2]

    def working_pressure_angle_with(self, other):
        """
        Working (operating) transverse pressure angle for the mesh.

        From ``inv(phi_wt) = inv(phi_t) + 2 x_eff tan(phi_n) / z_eff``,
        with the sums ``x1 + x2`` and ``z1 + z2`` for an external pair
        and the differences ring-minus-pinion for an internal one (see
        :func:`_mesh_terms`). Equals the transverse pressure angle when
        ``x_eff = 0``.

        Args:
            other (CylindricalGear): Mating gear.

        Returns:
            float: Working transverse pressure angle in degrees.

        Raises:
            ValueError: If the two gears cannot mesh.
        """
        from .transmission import _check_mesh

        _check_mesh(self, other)
        phi_t = math.radians(self.transverse_pressure_angle)
        phi_n = math.radians(self.pressure_angle)
        z_eff, x_eff, _, _ = _mesh_terms(self, other)
        inv_wt = involute(phi_t) + (2 * x_eff * math.tan(phi_n) / z_eff)
        return math.degrees(inverse_involute(inv_wt))

    def working_center_distance_with(self, other):
        """
        Working (mounting) center distance for a profile-shifted mesh.

        ``a_w = a_ref * cos(phi_t) / cos(phi_wt)``, using the reference
        center distance of :meth:`center_distance_with` (a difference
        for an internal mesh). Equals the reference center distance when
        ``x_eff = 0``.

        Args:
            other (CylindricalGear): Mating gear.

        Returns:
            float: Working center distance in mm.

        Raises:
            ValueError: If the two gears cannot mesh.
        """
        phi_t = math.radians(self.transverse_pressure_angle)
        phi_wt = math.radians(self.working_pressure_angle_with(other))
        a_ref = _mesh_terms(self, other)[2]
        return a_ref * math.cos(phi_t) / math.cos(phi_wt)

    def has_interference_with(self, other):
        """
        Check the mesh for involute (tip) interference.

        For an external mesh, interference occurs when a tip circle
        reaches inside the mate's base-tangent point on the line of
        action, i.e. for either member
        ``ra^2 > rb^2 + (a_w * sin(phi_wt))^2`` (Shigley). The check is
        symmetric and accounts for profile shift through the working
        center distance and pressure angle.

        For an internal mesh the external pinion is checked the same
        way, while the ring's tip must lie *beyond* the tangent point,
        so its condition is reversed
        (``ra^2 < rb^2 + (a_w sin(phi_wt))^2``). This covers involute
        interference only; see
        :meth:`has_trimming_interference_with` for the internal-specific
        trimming mode, which :meth:`describe` reports alongside it.

        Args:
            other (CylindricalGear): Mating gear.

        Returns:
            bool: True if the mesh has tip (involute) interference.

        Raises:
            ValueError: If the two gears cannot mesh.
        """
        phi_wt = math.radians(self.working_pressure_angle_with(other))
        a_w = self.working_center_distance_with(other)
        limit_sq = (a_w * math.sin(phi_wt)) ** 2
        for gear in (self, other):
            tangent_sq = gear.base_radius ** 2 + limit_sq
            if gear.internal:
                if gear.outside_radius ** 2 < tangent_sq - 1e-9:
                    return True
            elif gear.outside_radius ** 2 > tangent_sq + 1e-9:
                return True
        return False

    def has_trimming_interference_with(self, other):
        """
        Check an internal mesh for trimming (fouling) interference.

        When an external pinion is assembled into a ring gear with too
        few teeth of difference, the pinion tooth tips foul the ring
        tooth tips as the pinion is moved into place, trimming them.
        This uses the classical shop rule for 20 degree full-depth
        teeth — a minimum tooth difference of
        ``_MIN_INTERNAL_TOOTH_DIFFERENCE`` — and is an approximation of
        the full radial-insertion criterion, whose exact limit depends
        on the addendum system and the profile shifts.

        Args:
            other (CylindricalGear): Mating gear.

        Returns:
            bool: True if the mesh risks trimming interference. Always
            False for an external mesh, where this mode does not exist.

        Raises:
            ValueError: If the two gears cannot mesh.
        """
        from .transmission import _check_mesh

        _check_mesh(self, other)
        z_eff, _, _, internal = _mesh_terms(self, other)
        return internal and z_eff < _MIN_INTERNAL_TOOTH_DIFFERENCE

    def contact_ratio_with(self, other):
        """
        Transverse contact ratio for a mesh with another gear.

        Length of the line of action (at the working center distance)
        divided by the transverse base pitch. For an external pair both
        tip terms add and the center-distance term subtracts; for an
        internal pair the ring's tip term subtracts and the
        center-distance term adds. For helical gears use also
        :meth:`axial_contact_ratio_with` and
        :meth:`total_contact_ratio_with`.

        Args:
            other (CylindricalGear): Mating gear.

        Returns:
            float: Transverse contact ratio (dimensionless, > 1 for a
            correctly proportioned mesh).

        Raises:
            ValueError: If the two gears cannot mesh.
        """
        phi_wt = math.radians(self.working_pressure_angle_with(other))
        a_w = self.working_center_distance_with(other)
        internal = _mesh_terms(self, other)[3]
        center_term = a_w * math.sin(phi_wt)
        line_of_action = center_term if internal else -center_term
        for gear in (self, other):
            tip_term = math.sqrt(gear.outside_radius ** 2
                                 - gear.base_radius ** 2)
            line_of_action += -tip_term if gear.internal else tip_term
        return line_of_action / self.transverse_base_pitch

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

    def radial_force(self, tangential_force):
        """float: Radial (separating) force in N: Ft * tan(phi_t).

        Uses the transverse pressure angle, overriding
        :meth:`Gear.radial_force` (they coincide for a spur gear).
        ``tangential_force`` and :meth:`pitch_line_velocity` are
        inherited unchanged from :class:`~mecapy.gears.gear.Gear`.
        """
        phi_t = math.radians(self.transverse_pressure_angle)
        return tangential_force * math.tan(phi_t)

    def axial_force(self, tangential_force):
        """float: Axial (thrust) force in N: Ft * tan(beta)."""
        return tangential_force * math.tan(math.radians(self.helix_angle))

    def describe(self):
        """
        Multi-line summary of every geometry parameter.

        Extends :meth:`Gear.describe` with the transverse/helix block
        (helical gears only), face width and the undercut check. For an
        internal gear the undercut block is replaced by the minimum
        tooth difference that avoids trimming interference.

        Returns:
            str: Formatted geometry report.
        """
        lines = [super().describe()]
        lines.append(f"internal (ring) gear = "
                     f"{'yes' if self.internal else 'no'}")
        if self.helix_angle > 0:
            lines.append(
                f"helix angle (beta) = {self.helix_angle:.3f} deg")
            if self.hand is not None:
                lines.append(f"hand = {self.hand}")
            lines.append(
                f"transverse module (mt) = {self.transverse_module:.3f} mm")
            lines.append(
                f"transverse pressure angle (alpha_t) = "
                f"{self.transverse_pressure_angle:.3f} deg")
            lines.append(
                f"transverse base pitch (pbt) = "
                f"{self.transverse_base_pitch:.3f} mm")
            lines.append(f"axial pitch (px) = {self.axial_pitch:.3f} mm")
            lines.append(f"lead (pz) = {self.lead:.3f} mm")
        if self.face_width is not None:
            lines.append(f"face width (b) = {self.face_width:.3f} mm")
        if self.internal:
            lines.append(
                f"min tooth difference vs pinion (dz) = "
                f"{_MIN_INTERNAL_TOOTH_DIFFERENCE} (trimming limit)")
        else:
            lines.append(
                f"min teeth without undercut (Zmin) = "
                f"{self.min_teeth_no_undercut}")
            lines.append(
                f"min profile shift (x_min) = {self.min_profile_shift:.3f}")
            lines.append(f"undercut = {'yes' if self.is_undercut else 'no'}")
        return "\n".join(lines)

    def rate_agma(self, mate, **kwargs):
        """
        AGMA bending and pitting rating of this gear (pinion) with a mate.

        Convenience wrapper around
        :class:`~mecapy.gears.agma.AGMARating`; ``self`` is taken as the
        pinion (driver). Note: the AGMA J-factor tables assume standard
        (x = 0) tooth proportions; ratings for profile-shifted gears
        are approximate.

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
                 face_width=None, profile_shift=0.0, internal=False,
                 material="steel", name=None, diametral_pitch=None,
                 power_kw=None, speed_rpm=None):
        """
        Initialize a spur gear.

        Args:
            teeth (int): Number of teeth (>= 4).
            module (float): Module in mm (or give ``diametral_pitch``).
            pressure_angle (float): Pressure angle in degrees (default 20).
            face_width (float): Face width in mm (needed for AGMA rating).
            profile_shift (float): Profile shift coefficient x,
                -1 < x < 1 (default 0 = standard teeth).
            internal (bool): True for an internal (ring) gear
                (default: False).
            material (str): Material type (default: "steel").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to module.
        """
        super().__init__(teeth, module=module, pressure_angle=pressure_angle,
                         helix_angle=0.0, hand=None, face_width=face_width,
                         profile_shift=profile_shift, internal=internal,
                         material=material,
                         name=name, diametral_pitch=diametral_pitch,
                         power_kw=power_kw, speed_rpm=speed_rpm)


class HelicalGear(CylindricalGear):
    """
    Helical gear: teeth at a helix angle to the axis.

    ``module`` and ``pressure_angle`` are normal-plane values; two
    external helical gears mesh on parallel axes when they share module,
    pressure angle, helix angle and have *opposite* hands. An internal
    helical mesh is the exception: there the two members need the
    *same* hand.
    """

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 helix_angle=None, hand=None, face_width=None,
                 profile_shift=0.0, internal=False, material="steel",
                 name=None, diametral_pitch=None, power_kw=None,
                 speed_rpm=None):
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
            profile_shift (float): Profile shift coefficient x,
                -1 < x < 1 (default 0 = standard teeth).
            internal (bool): True for an internal (ring) gear
                (default: False).
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
                         face_width=face_width, profile_shift=profile_shift,
                         internal=internal,
                         material=material, name=name,
                         diametral_pitch=diametral_pitch,
                         power_kw=power_kw, speed_rpm=speed_rpm)


class HerringboneGear(HelicalGear):
    """
    Herringbone (double-helical) gear: two opposed helical halves.

    The opposed halves cancel axial thrust, so the gear has no hand and
    ``net_axial_force`` is zero. ``face_width`` is the total width of
    both halves. Meshes with another herringbone gear of equal helix
    angle.
    """

    def __init__(self, teeth, module=None, pressure_angle=20.0,
                 helix_angle=None, face_width=None, profile_shift=0.0,
                 internal=False, material="steel", name=None,
                 diametral_pitch=None, power_kw=None, speed_rpm=None):
        """
        Initialize a herringbone gear.

        Args:
            teeth (int): Number of teeth (>= 4).
            module (float): Normal module in mm (or ``diametral_pitch``).
            pressure_angle (float): Normal pressure angle in degrees.
            helix_angle (float): Helix angle of each half in degrees,
                0 < angle <= 45. Required.
            face_width (float): Total face width of both halves in mm.
            profile_shift (float): Profile shift coefficient x,
                -1 < x < 1 (default 0 = standard teeth).
            internal (bool): True for an internal (ring) gear
                (default: False).
            material (str): Material type (default: "steel").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to module.
        """
        super().__init__(teeth, module=module, pressure_angle=pressure_angle,
                         helix_angle=helix_angle, hand="right",
                         face_width=face_width, profile_shift=profile_shift,
                         internal=internal,
                         material=material, name=name,
                         diametral_pitch=diametral_pitch,
                         power_kw=power_kw, speed_rpm=speed_rpm)
        # The opposed halves have no net hand.
        self.hand = None

    @property
    def net_axial_force(self):
        """float: Net axial force in N — always 0 (thrust cancels)."""
        return 0.0

    def axial_force(self, tangential_force):
        """float: Net axial force in N — always 0 for herringbone."""
        return 0.0
