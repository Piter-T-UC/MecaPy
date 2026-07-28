"""AGMA bending and pitting resistance rating (metric, 2101-D04 form).

Implements the AGMA fundamental stress equations for external
cylindrical gear meshes (spur / helical / herringbone, including a
pinion on a rack), as presented in Shigley's Mechanical Engineering
Design ch. 14:

    bending:  sigma_F = Ft Ko Kv Ks (1 / (b mt)) (KH KB / YJ)
    contact:  sigma_H = ZE sqrt(Ft Ko Kv Ks (KH / (dw1 b)) (ZR / ZI))

with allowable stresses

    sigma_FP = St YN / (Y_theta YZ)      SF = sigma_FP / sigma_F
    sigma_HP = Sc ZN ZW / (Y_theta YZ)   SH = sigma_HP / sigma_H

Units: N, mm, MPa, m/s. Bevel and worm gears are NOT covered (their
AGMA formulations differ); see :mod:`mecapy.gears.bevel` and
:mod:`mecapy.gears.worm` for the simplified checks provided instead.

Internal (ring) meshes reuse the same equations with internal contact
geometry — the concave ring flank in
:func:`geometry_factor_I`, ZW forced to 1 — and are **approximate**:
the J-factor tables are external-tooth data, and AGMA's own internal
provisions are not implemented. Treat them as design exploration, in
the same register as the bevel and worm checks.
"""

import math

from . import agma_data
from .material import GearMaterial
from .rack import Rack


# ----------------------------------------------------------------------
# Modification-factor functions
# ----------------------------------------------------------------------

def dynamic_factor(pitch_line_velocity, quality_number=7):
    """
    AGMA dynamic factor Kv from the transmission quality number.

    ``B = 0.25 (12 - Qv)^(2/3)``, ``A = 50 + 56 (1 - B)``,
    ``Kv = ((A + sqrt(200 v)) / A)^B`` with v in m/s.

    Args:
        pitch_line_velocity (float): Pitch-line velocity v in m/s.
        quality_number (int): AGMA quality number Qv, 3 to 11
            (default: 7, typical commercial gearing).

    Returns:
        float: Dynamic factor Kv (>= 1).

    Raises:
        ValueError: If Qv is outside 3..11 or the velocity exceeds the
            validity limit ``(A + (Qv - 3))^2 / 200``.
    """
    if not 3 <= quality_number <= 11:
        raise ValueError("Quality number Qv must be between 3 and 11")
    if pitch_line_velocity < 0:
        raise ValueError("Pitch-line velocity must be non-negative")
    b = 0.25 * (12 - quality_number) ** (2.0 / 3.0)
    a = 50 + 56 * (1 - b)
    v_max = (a + (quality_number - 3)) ** 2 / 200.0
    if pitch_line_velocity > v_max:
        raise ValueError(
            f"Velocity {pitch_line_velocity:.2f} m/s exceeds the Kv "
            f"validity limit {v_max:.2f} m/s for Qv={quality_number}"
        )
    return ((a + math.sqrt(200 * pitch_line_velocity)) / a) ** b


def load_distribution_factor(face_width, pitch_diameter,
                             condition="commercial", crowned=False):
    """
    AGMA load-distribution factor KH (Km).

    ``KH = 1 + Cmc (Cpf Cpm + Cma Ce)`` with Cpf from the face width and
    pinion diameter (Shigley Eq. 14-32) and Cma from the Table 14-9
    curve fits. Cpm and Ce are taken as 1 (typical straddle-mounted,
    non-lapped gearing).

    Args:
        face_width (float): Face width b in mm.
        pitch_diameter (float): Pinion pitch diameter in mm.
        condition (str): Gearing enclosure/accuracy condition — "open",
            "commercial" (default), "precision" or "extra_precision".
        crowned (bool): True for crowned teeth (Cmc = 0.8).

    Returns:
        float: Load-distribution factor KH.

    Raises:
        ValueError: For non-physical inputs or an unknown condition.
    """
    if face_width <= 0 or pitch_diameter <= 0:
        raise ValueError("Face width and pitch diameter must be positive")
    if condition not in agma_data.CMA_COEFFICIENTS:
        raise ValueError(
            f"Unknown condition {condition!r}; available: "
            f"{sorted(agma_data.CMA_COEFFICIENTS)}"
        )
    cmc = 0.8 if crowned else 1.0
    # Eq. 14-32 uses US units: F and d in inches.
    f_in = face_width / 25.4
    d_in = pitch_diameter / 25.4
    ratio = f_in / (10 * d_in)
    if ratio < 0.05:
        ratio = 0.05
    if f_in <= 1:
        cpf = ratio - 0.025
    elif f_in <= 17:
        cpf = ratio - 0.0375 + 0.0125 * f_in
    elif f_in <= 40:
        cpf = ratio - 0.1109 + 0.0207 * f_in - 0.000228 * f_in ** 2
    else:
        raise ValueError("Face width beyond the Cpf fit range (40 in)")
    a, b_coef, c = agma_data.CMA_COEFFICIENTS[condition]
    cma = a + b_coef * f_in + c * f_in ** 2
    return 1 + cmc * (cpf * 1.0 + cma * 1.0)


def size_factor(module):
    """
    AGMA size factor Ks from the tooth size (module).

    Stepped metric table — Ks is 1 for teeth up to module 5 and grows
    with tooth size, penalising the lower material strength of a large
    section::

        m <= 5    Ks = 1.00
        m <= 6    Ks = 1.05
        m <= 8    Ks = 1.15
        m <= 12   Ks = 1.25
        m <= 20   Ks = 1.40

    Each listed module is the UPPER bound of its band, so m = 6 gives
    1.05 and m = 8 gives 1.15. Above module 20 the table is clamped at
    1.40 (the fit is not defined further).

    Use the NORMAL module for a helical gear: Ks follows the physical
    tooth size, not the transverse section.

    Args:
        module (float): Normal module m in mm (> 0).

    Returns:
        float: Size factor Ks (>= 1).

    Raises:
        ValueError: If the module is not strictly positive.
    """
    if module <= 0:
        raise ValueError("Module must be strictly positive")
    for bound, ks in agma_data.SIZE_FACTOR_BY_MODULE:
        if module <= bound:
            return ks
    return agma_data.SIZE_FACTOR_BY_MODULE[-1][1]


def rim_thickness_factor(backup_ratio):
    """
    AGMA rim-thickness factor KB.

    ``KB = 1.6 ln(2.242 / mB)`` for backup ratio mB < 1.2, else 1.
    The backup ratio is the rim thickness below the root divided by the
    whole tooth depth.

    Args:
        backup_ratio (float): Backup ratio mB (> 0).

    Returns:
        float: Rim-thickness factor KB (>= 1).

    Raises:
        ValueError: If the backup ratio is not strictly positive.
    """
    if backup_ratio <= 0:
        raise ValueError("Backup ratio must be strictly positive")
    if backup_ratio >= 1.2:
        return 1.0
    return 1.6 * math.log(2.242 / backup_ratio)


def elastic_coefficient(pinion_properties, gear_properties):
    """
    AGMA elastic coefficient ZE (Cp).

    ``ZE = sqrt(1 / (pi ((1 - nu1^2)/E1 + (1 - nu2^2)/E2)))`` with the
    elastic moduli in MPa, giving ZE in sqrt(MPa) (about 190 for a
    steel-steel mesh).

    Args:
        pinion_properties (dict): Material dict with "elastic_modulus"
            (Pa) and "poisson_ratio".
        gear_properties (dict): Same for the gear.

    Returns:
        float: Elastic coefficient in sqrt(MPa).
    """
    e1 = pinion_properties["elastic_modulus"] / 1e6  # Pa -> MPa
    e2 = gear_properties["elastic_modulus"] / 1e6
    nu1 = pinion_properties["poisson_ratio"]
    nu2 = gear_properties["poisson_ratio"]
    return math.sqrt(1.0 / (math.pi * ((1 - nu1 ** 2) / e1
                                       + (1 - nu2 ** 2) / e2)))


def geometry_factor_I(pinion, gear=None):
    """
    Surface (pitting) geometry factor I per Norton, *Machine Design*.

    ``I = cos(phi_t) / ((1/rho_p + 1/rho_g) dp)`` (Norton Eq. 12.22,
    metric form) with the flank curvature radii taken one transverse
    base pitch inside the pinion tip (lowest point of single-tooth
    contact)::

        rho_p = sqrt(ra_p^2 - rb_p^2) - pi mt cos(phi_t)
        rho_g = C sin(phi_t) - rho_p

    Profile shift enters through the pinion outside radius (addendum
    m(1 + x)) and the working center distance C. Helical gears are
    handled in the transverse plane (phi_t, mt). A rack flank is
    straight, so ``gear=None`` uses ``1 / rho_g = 0``.

    An internal (ring) gear has a *concave* flank, so its curvature
    subtracts instead of adding::

        rho_g = rho_p + C sin(phi_t)
        I = cos(phi_t) / ((1/rho_p - 1/rho_g) dp)

    which reproduces the closed-form
    ``I = cos(phi_t) sin(phi_t) mG / (2 (mG - 1))`` — larger than the
    external ``mG / (mG + 1)`` form, i.e. an internal mesh has lower
    contact stress at the same load, as expected.

    Args:
        pinion (CylindricalGear): The pinion — always the external
            member of an internal mesh.
        gear (CylindricalGear): The mating gear; ``None`` for a pinion
            driving a rack.

    Returns:
        float: Geometry factor I (ZI, dimensionless).

    Raises:
        ValueError: If contact falls below the pinion base circle
            (rho_p <= 0) or an external mesh has tip interference
            (rho_g <= 0).
    """
    phi_t = math.radians(pinion.transverse_pressure_angle)
    ra = pinion.outside_radius
    rb = pinion.base_radius
    base_pitch = math.pi * pinion.transverse_module * math.cos(phi_t)
    rho_p = math.sqrt(ra ** 2 - rb ** 2) - base_pitch
    if rho_p <= 0:
        raise ValueError("Pinion contact falls below the base circle "
                         "(rho_p <= 0); too few teeth or undercut")
    if gear is None:
        curvature = 1.0 / rho_p
    elif gear.internal:
        c = pinion.working_center_distance_with(gear)
        rho_g = rho_p + c * math.sin(phi_t)
        curvature = 1.0 / rho_p - 1.0 / rho_g
    else:
        c = pinion.working_center_distance_with(gear)
        rho_g = c * math.sin(phi_t) - rho_p
        if rho_g <= 0:
            raise ValueError("Gear flank curvature is non-positive "
                             "(rho_g <= 0); mesh has tip interference")
        curvature = 1.0 / rho_p + 1.0 / rho_g
    return math.cos(phi_t) / (curvature * pinion.pitch_diameter)


def bending_life_factor(cycles=1e7):
    """
    AGMA bending stress-cycle factor YN (Fig. 14-14 upper fit).

    ``YN = 1.3558 N^-0.0178`` (about 1.0 at 10^7 cycles). Valid for
    roughly 10^3 to 10^10 cycles.

    Args:
        cycles (float): Number of load cycles N (default: 1e7).

    Returns:
        float: Stress-cycle factor YN.

    Raises:
        ValueError: If ``cycles`` is not strictly positive.
    """
    if cycles <= 0:
        raise ValueError("Cycle count must be strictly positive")
    return 1.3558 * cycles ** -0.0178


def contact_life_factor(cycles=1e7):
    """
    AGMA pitting stress-cycle factor ZN (Fig. 14-15 fit).

    ``ZN = 1.4488 N^-0.023`` (about 1.0 at 10^7 cycles) for steel.

    Args:
        cycles (float): Number of load cycles N (default: 1e7).

    Returns:
        float: Stress-cycle factor ZN.

    Raises:
        ValueError: If ``cycles`` is not strictly positive.
    """
    if cycles <= 0:
        raise ValueError("Cycle count must be strictly positive")
    elif cycles < 1e4:
        return 1.5
    elif cycles < 1e7:
        return 2.466 * cycles ** -0.056
    return 1.4488 * cycles ** -0.023


def temperature_factor(temperature_c):
    """
    AGMA temperature (derating) factor Y_theta from a temperature.

    ``Y_theta = 1.0`` for oil temperature T <= 110 C; above 110 C,
    ``Y_theta = (220 + T) / 330``. Use the result as a divisor in the
    allowable-stress calculation to derate strength at elevated
    temperatures (e.g. ``allowable = base / temperature_factor``).

    Args:
        temperature_c (float): Operating (oil) temperature in degrees
            Celsius.

    Returns:
        float: Temperature factor Y_theta (>= 1).

    Raises:
        ValueError: If the temperature is below absolute zero.
    """
    if temperature_c < -273.15:
        raise ValueError("Temperature must be above absolute zero (-273.15 C)")
    if temperature_c < 110.0:
        return 1.0
    return (220.0 + temperature_c) / 330.0


_temperature_factor_from_temp = temperature_factor


def hardness_ratio_factor(pinion_hardness, gear_hardness, gear_ratio):
    """
    AGMA hardness-ratio factor ZW (CH), applied to the GEAR only.

    ``ZW = 1 + A' (mG - 1)`` with A' from the pinion/gear Brinell
    hardness ratio (Shigley Eq. 14-36).

    Args:
        pinion_hardness (float): Pinion Brinell hardness HB.
        gear_hardness (float): Gear Brinell hardness HB.
        gear_ratio (float): mG = gear teeth / pinion teeth.

    Returns:
        float: Hardness-ratio factor ZW (>= 1).
    """
    ratio = pinion_hardness / gear_hardness
    if ratio < 1.2:
        a_prime = 0.0
    elif ratio <= 1.7:
        a_prime = 8.98e-3 * ratio - 8.29e-3
    else:
        a_prime = 6.98e-3
    return 1 + a_prime * (gear_ratio - 1)


# ----------------------------------------------------------------------
# Rating
# ----------------------------------------------------------------------

class AGMARating:
    """
    AGMA bending and pitting rating of an external cylindrical mesh.

    Inputs are validated and stored at construction; every modification
    factor, stress and safety factor is a lazy ``@property`` recomputed
    from the pinion, gear and rating inputs on each access, so mutating a
    gear (teeth, face width, material) or a rating factor (Ko, Ks, ...) is
    reflected immediately — nothing derived is ever cached and stale. The
    first gear is the pinion (driver); the mate may be a gear or a
    :class:`Rack`.

    The ``YJ_pinion``/``YJ_gear`` overrides are used verbatim when given
    (not recomputed); otherwise the geometry factor is computed from the
    tooth counts and helix angle. ``St``/``Sc`` behave the same way.

    Note: the digitized J-factor and Lewis tables assume standard
    (profile shift x = 0) tooth proportions; ratings for
    profile-shifted gears are approximate.

    Attributes:
        Ft (float): Tangential force in N.
        pitch_line_velocity (float): In m/s.
        Kv, KH, KB, Ks, Ko (float): Modification factors.
        Ki_pinion, Ki_gear (float): Idler bending factors (1.42 for a
            member that meshes in more than one stage, else 1.0).
        ZI, ZE (float): Surface geometry and elastic coefficients.
        YJ_pinion, YJ_gear (float): Bending geometry factors.
        bending_stress_pinion, bending_stress_gear (float): MPa.
        contact_stress (float): MPa (same for both members).
        allowable_bending_stress, allowable_contact_stress (float): MPa.
        SF_pinion, SF_gear (float): Bending safety factors.
        SH (float): Pitting safety factor (compare SH^2 with SF for an
            equal-confidence comparison).
    """

    def __init__(self, pinion, gear, power_kw=None, pinion_speed_rpm=None,
                 tangential_force=None, Ko=1.0, Qv=7, Ks=None, ZR=1.0,
                 KB=1.0, Ki_pinion=1.0, Ki_gear=1.0, life_cycles=1e7,
                 reliability=0.99, grade=1, hardness_HB=None,
                 gear_hardness_HB=None, St=None, Sc=None, YJ_pinion=None,
                 YJ_gear=None, condition="commercial", crowned=False,
                 temperature_factor=1.0, temperature_celsius=60):
        """
        Evaluate the AGMA rating of a pinion-gear (or pinion-rack) mesh.

        Args:
            pinion (CylindricalGear): Driving pinion. ``face_width``
                must be set.
            gear (CylindricalGear or Rack): Mating gear or rack.
            power_kw (float): Transmitted power in kW (give together
                with ``pinion_speed_rpm``, or give ``tangential_force``).
            pinion_speed_rpm (float): Pinion speed in rpm. Required.
            tangential_force (float): Tangential force Ft in N
                (alternative to ``power_kw``).
            Ko (float): Overload factor (default 1.0). Typical values:
                uniform/uniform 1.0, light shock 1.25, moderate shock
                1.5, heavy shock 1.75+.
            Qv (int): AGMA quality number for Kv (default 7).
            Ks (float): Size factor. Default ``None`` computes it from
                the pinion's normal module via :func:`size_factor`
                (Ks = 1.0 up to module 5, rising to 1.40 at module 20).
                Give a number to override the table.
            ZR (float): Surface-condition factor (default 1.0).
            KB (float): Rim-thickness factor (default 1.0, solid gear);
                see :func:`rim_thickness_factor`. Always caller-supplied;
                on a ring gear the backup ratio is measured on the rim
                *outside* the root circle, which is a design choice this
                class has no way to know.
            Ki_pinion (float): Idler bending factor for the pinion member
                (default 1.0). An idler tooth is loaded on both flanks
                each revolution (fully reversed bending instead of
                repeated), so it needs roughly 1/0.70 = 1.42 times the
                one-directional bending stress. :meth:`Transmission.rate_agma`
                sets this automatically per stage when the pinion object
                also appears in another stage of the train.
            Ki_gear (float): Same idler bending factor, for the gear
                member (default 1.0).
            life_cycles (float): Load cycles for YN/ZN (default 1e7).
            reliability (float): Survival probability for YZ
                (default 0.99).
            grade (int): AGMA steel grade for the allowable-stress fits
                (default 1). Ignored when the hardness is taken from a
                :class:`~mecapy.gears.GearMaterial`, which supplies its
                own grade.
            hardness_HB (float): Brinell hardness for the through-
                hardened allowable-stress fits. Give this OR explicit
                ``St``/``Sc``. Defaults to the pinion material's
                ``hardness_HB`` when it is a
                :class:`~mecapy.gears.GearMaterial`.
            gear_hardness_HB (float): Gear hardness if different from
                the pinion (enables the ZW factor). Defaults to the gear
                material's ``hardness_HB`` when it is a
                :class:`~mecapy.gears.GearMaterial`.
            St (float): Explicit allowable bending stress number in MPa.
            Sc (float): Explicit allowable contact stress number in MPa.
            YJ_pinion (float): Override for the pinion bending geometry
                factor J.
            YJ_gear (float): Override for the gear J.
            condition (str): Mounting condition for KH
                (default "commercial").
            crowned (bool): Crowned teeth for KH (default False).
            temperature_factor (float): Y_theta (default 1.0, valid for
                oil temperatures up to about 120 C). Give this OR
                ``temperature_celsius``, not both.
            temperature_celsius (float): Operating temperature in degrees
                Celsius; converted to Y_theta via :func:`temperature_factor`.
                Mutually exclusive with ``temperature_factor``.

        Raises:
            ValueError: If the mesh is incompatible, face widths are
                missing, the load specification is ambiguous, no
                material allowable can be determined, ``Ki_pinion`` or
                ``Ki_gear`` is not strictly positive, or both (or
                neither) temperature inputs are given.
        """
        from .transmission import _check_mesh

        _check_mesh(pinion, gear)
        if isinstance(pinion, Rack):
            raise ValueError("The pinion (first argument) cannot be a rack")
        if pinion.face_width is None:
            raise ValueError("Pinion face width must be set for AGMA rating")
        if pinion_speed_rpm is None or pinion_speed_rpm <= 0:
            raise ValueError("A positive pinion speed (rpm) is required")
        if (power_kw is None) == (tangential_force is None):
            raise ValueError(
                "Give exactly one of 'power_kw' or 'tangential_force'"
            )
        if tangential_force is not None and tangential_force <= 0:
            raise ValueError("Tangential force must be strictly positive")
        if Ki_pinion <= 0:
            raise ValueError("Ki_pinion must be strictly positive")
        if Ki_gear <= 0:
            raise ValueError("Ki_gear must be strictly positive")
        # No eager material check: the stresses need no allowable, so a
        # rating built without one is valid and usable for material
        # selection. St/Sc raise only when actually accessed.
        if (temperature_factor != 1.0) and (temperature_celsius is not None):
            raise ValueError(
                "Give 'temperature_factor' (explicit Y_theta) or "
                "'temperature_celsius' (oil temperature), not both"
            )
        if temperature_celsius is not None:
            temperature_factor = _temperature_factor_from_temp(temperature_celsius)
        if pinion.internal:
            raise ValueError(
                "The pinion (first argument) must be the external member; "
                "pass the internal (ring) gear as 'gear'"
            )
        if not isinstance(gear, Rack) and gear.teeth < pinion.teeth:
            raise ValueError(
                "The pinion must be the smaller member (gear ratio >= 1)"
            )

        # Store inputs only; everything derived is a lazy property below.
        self.pinion = pinion
        self.gear = gear
        self.Ko = Ko
        self._Ks = Ks
        self.ZR = ZR
        self.KB = KB
        self.Ki_pinion = Ki_pinion
        self.Ki_gear = Ki_gear
        self.temperature_factor = temperature_factor
        self._Qv = Qv
        self._life_cycles = life_cycles
        self._reliability = reliability
        self._condition = condition
        self._crowned = crowned
        self._hardness_HB = hardness_HB
        self._grade = grade
        self._gear_hardness_HB = gear_hardness_HB
        self._St = St
        self._Sc = Sc
        self._YJ_pinion = YJ_pinion
        self._YJ_gear = YJ_gear
        self._power_kw = power_kw
        self._pinion_speed_rpm = pinion_speed_rpm
        self._tangential_force = tangential_force

    # ---- Geometry / load (recomputed, never cached) ----

    @property
    def is_rack(self):
        """bool: Whether the mating member is a :class:`Rack`."""
        return isinstance(self.gear, Rack)

    @property
    def gear_ratio(self):
        """float: mG = gear teeth / pinion teeth (1e6 for a rack)."""
        if self.is_rack:
            return 1e6
        ratio = self.gear.teeth / self.pinion.teeth
        if ratio < 1:
            raise ValueError(
                "The pinion must be the smaller member (gear ratio >= 1)"
            )
        return ratio

    @property
    def face_width(self):
        """float: Effective face width b in mm (the narrower member)."""
        b = self.pinion.face_width
        if not self.is_rack and self.gear.face_width is not None:
            b = min(b, self.gear.face_width)
        return b

    @property
    def pitch_line_velocity(self):
        """float: Pitch-line velocity in m/s."""
        return self.pinion.pitch_line_velocity(self._pinion_speed_rpm)

    @property
    def Ft(self):
        """float: Tangential force in N (given, or from power and speed)."""
        if self._tangential_force is not None:
            return self._tangential_force
        return self.pinion.tangential_force(self._power_kw,
                                            self._pinion_speed_rpm)

    # ---- Modification factors ----

    @property
    def Kv(self):
        """float: Dynamic factor (see :func:`dynamic_factor`)."""
        return dynamic_factor(self.pitch_line_velocity, self._Qv)

    @property
    def Ks(self):
        """float: Size factor (see :func:`size_factor`), from the pinion's
        normal module unless an explicit value was given. Recomputed on
        access, so changing the gear's module updates it.
        """
        if self._Ks is not None:
            return self._Ks
        return size_factor(self.pinion.module)

    @Ks.setter
    def Ks(self, value):
        if value is not None and value <= 0:
            raise ValueError("Size factor Ks must be strictly positive")
        self._Ks = value

    @property
    def KH(self):
        """float: Load-distribution factor (see
        :func:`load_distribution_factor`)."""
        return load_distribution_factor(
            self.face_width, self.pinion.pitch_diameter,
            condition=self._condition, crowned=self._crowned)

    @property
    def YJ_pinion(self):
        """float: Pinion bending geometry factor (override if given)."""
        if self._YJ_pinion is not None:
            return self._YJ_pinion
        mating = "rack" if self.is_rack else self.gear.teeth
        return agma_data.geometry_factor_J(
            self.pinion.teeth, mating, self.pinion.helix_angle)

    @property
    def YJ_gear(self):
        """float or None: Gear bending geometry factor (None for a rack)."""
        if self.is_rack:
            return None
        if self._YJ_gear is not None:
            return self._YJ_gear
        return agma_data.geometry_factor_J(
            self.gear.teeth, self.pinion.teeth, self.gear.helix_angle)

    @property
    def ZI(self):
        """float: Surface (pitting) geometry factor (see
        :func:`geometry_factor_I`)."""
        return geometry_factor_I(self.pinion,
                                 gear=None if self.is_rack else self.gear)

    @property
    def ZE(self):
        """float: Elastic coefficient in sqrt(MPa)."""
        gear_props = (self.pinion.material_properties if self.is_rack
                      else self.gear.material_properties)
        return elastic_coefficient(self.pinion.material_properties, gear_props)

    @property
    def YN(self):
        """float: Bending stress-cycle factor."""
        return bending_life_factor(self._life_cycles)

    @property
    def ZN(self):
        """float: Pitting stress-cycle factor."""
        return contact_life_factor(self._life_cycles)

    @property
    def YZ(self):
        """float: Reliability factor."""
        return agma_data.reliability_factor(self._reliability)

    @property
    def _hardness(self):
        """float or None: Pinion Brinell hardness (arg, else GearMaterial)."""
        if self._hardness_HB is not None:
            return self._hardness_HB
        mat = self.pinion.material
        return mat.hardness_HB if isinstance(mat, GearMaterial) else None

    @property
    def _grade_resolved(self):
        """int: AGMA grade (the GearMaterial's when hardness comes from it)."""
        if self._hardness_HB is None and isinstance(self.pinion.material,
                                                    GearMaterial):
            return self.pinion.material.grade
        return self._grade

    @property
    def _gear_hardness(self):
        """float or None: Gear Brinell hardness (arg, else GearMaterial)."""
        if self._gear_hardness_HB is not None:
            return self._gear_hardness_HB
        if not self.is_rack and isinstance(self.gear.material, GearMaterial):
            return self.gear.material.hardness_HB
        return None

    @property
    def is_internal_mesh(self):
        """bool: Whether the mating member is an internal (ring) gear."""
        return bool(getattr(self.gear, "internal", False))

    @property
    def ZW(self):
        """float: Hardness-ratio factor ZW (1.0 unless both hardnesses set).

        AGMA defines ZW for external meshes only, so it is forced to 1.0
        on an internal mesh.
        """
        if self.is_internal_mesh:
            return 1.0
        ph, gh = self._hardness, self._gear_hardness
        if ph is not None and gh is not None:
            return hardness_ratio_factor(ph, gh, self.gear_ratio)
        return 1.0

    @property
    def has_allowables(self):
        """bool: Whether allowable stresses (and therefore safety factors)
        can be computed — True when a hardness is available (given or from
        a :class:`~mecapy.gears.GearMaterial`) or both St and Sc are given.

        False means the stresses are still valid; only the strength side
        is missing, which is the normal state when rating a mesh in order
        to *choose* a material.
        """
        if self._hardness is not None:
            return True
        return self._St is not None and self._Sc is not None

    def _require_allowables(self, symbol):
        """Raise a clear ValueError when no material strength is available."""
        raise ValueError(
            f"Cannot compute {symbol}: no material strength data. Give "
            f"'hardness_HB' (through-hardened steel fits), a GearMaterial "
            f"on the gears, or explicit 'St' and 'Sc' in MPa. The stresses "
            f"themselves need none of this."
        )

    @property
    def St(self):
        """float: Allowable bending stress number in MPa (override if given).

        Raises:
            ValueError: If no hardness or explicit St is available.
        """
        if self._St is not None:
            return self._St
        if self._hardness is None:
            self._require_allowables("St")
        return agma_data.allowable_bending_stress(self._hardness,
                                                  self._grade_resolved)

    @property
    def Sc(self):
        """float: Allowable contact stress number in MPa (override if given).

        Raises:
            ValueError: If no hardness or explicit Sc is available.
        """
        if self._Sc is not None:
            return self._Sc
        if self._hardness is None:
            self._require_allowables("Sc")
        return agma_data.allowable_contact_stress(self._hardness,
                                                  self._grade_resolved)

    # ---- Stresses and safety factors ----

    @property
    def _common_load(self):
        """float: Ft*Ko*Kv*Ks, shared by the bending and contact stresses."""
        return self.Ft * self.Ko * self.Kv * self.Ks

    @property
    def bending_stress_pinion(self):
        """float: Pinion bending stress sigma_F in MPa (includes
        ``Ki_pinion`` when the pinion is an idler)."""
        return (self._common_load / (self.face_width
                                     * self.pinion.transverse_module)
                * self.KH * self.KB / self.YJ_pinion) * self.Ki_pinion

    @property
    def bending_stress_gear(self):
        """float or None: Gear bending stress in MPa (None for a rack;
        includes ``Ki_gear`` when the gear is an idler)."""
        yj = self.YJ_gear
        if yj is None:
            return None
        return (self._common_load / (self.face_width
                                     * self.pinion.transverse_module)
                * self.KH * self.KB / yj) * self.Ki_gear

    @property
    def contact_stress(self):
        """float: Contact (pitting) stress sigma_H in MPa (both members)."""
        return self.ZE * math.sqrt(
            self._common_load * self.KH
            / (self.pinion.pitch_diameter * self.face_width)
            * self.ZR / self.ZI
        )

    @property
    def allowable_bending_stress(self):
        """float: Allowable bending stress sigma_FP in MPa."""
        return self.St * self.YN / (self.temperature_factor * self.YZ)

    @property
    def allowable_contact_stress(self):
        """float: Allowable contact stress sigma_HP in MPa."""
        return (self.Sc * self.ZN * self.ZW
                / (self.temperature_factor * self.YZ))

    @property
    def SF_pinion(self):
        """float: Pinion bending safety factor."""
        return self.allowable_bending_stress / self.bending_stress_pinion

    @property
    def SF_gear(self):
        """float or None: Gear bending safety factor (None for a rack)."""
        sg = self.bending_stress_gear
        if sg is None:
            return None
        return self.allowable_bending_stress / sg

    @property
    def SH(self):
        """float: Pitting safety factor (compare SH^2 with SF)."""
        return self.allowable_contact_stress / self.contact_stress

    def summary(self):
        """
        Human-readable rating summary.

        Returns:
            str: Multi-line summary of loads, factors, stresses and
            safety factors.
        """
        lines = [
            "AGMA Rating Summary",
            "=" * 40,
            f"Pinion: {self.pinion!r}",
            f"Gear:   {self.gear!r}",
            f"Tangential force Ft:     {self.Ft:.1f} N",
            f"Pitch-line velocity:     {self.pitch_line_velocity:.2f} m/s",
            f"Factors: Ko={self.Ko:.2f} Kv={self.Kv:.3f} Ks={self.Ks:.2f} "
            f"KH={self.KH:.3f} KB={self.KB:.2f}"
            + (f" Ki_pinion={self.Ki_pinion:.2f}" if self.Ki_pinion != 1.0 else "")
            + (f" Ki_gear={self.Ki_gear:.2f}" if self.Ki_gear != 1.0 else ""),
            f"Geometry: YJ_pinion={self.YJ_pinion:.3f} "
            + (f"YJ_gear={self.YJ_gear:.3f} " if self.YJ_gear else "")
            + f"ZI={self.ZI:.4f} ZE={self.ZE:.1f} sqrt(MPa)",
            f"Bending stress (pinion): {self.bending_stress_pinion:.1f} MPa",
        ]
        if self.bending_stress_gear is not None:
            lines.append(
                f"Bending stress (gear):   {self.bending_stress_gear:.1f} MPa"
            )
        lines.append(f"Contact stress:          {self.contact_stress:.1f} MPa")
        if self.has_allowables:
            lines += [
                f"Allowable bending:       "
                f"{self.allowable_bending_stress:.1f} MPa",
                f"Allowable contact:       "
                f"{self.allowable_contact_stress:.1f} MPa",
                f"Safety factors: SF_pinion={self.SF_pinion:.2f} "
                + (f"SF_gear={self.SF_gear:.2f} " if self.SF_gear else "")
                + f"SH={self.SH:.2f}",
            ]
        else:
            lines += [
                "No material strength given - stresses only.",
                f"For a target SF, pick a material with "
                f"St >= {self.required_St():.1f} MPa and "
                f"Sc >= {self.required_Sc():.1f} MPa (SF = SH = 1).",
            ]
        return "\n".join(lines)

    def required_St(self, safety_factor=1.0, max_safety_factor=None):
        """
        Allowable bending stress number a material must have.

        Inverts ``SF = St YN / (Y_theta YZ sigma_F)`` for St, using the
        larger of the pinion and gear bending stresses. Needs no material
        data, so it is the natural way to go from a computed stress to a
        material choice.

        Args:
            safety_factor (float): Target (minimum) bending safety factor
                SF (default 1.0).
            max_safety_factor (float): Optional upper bound on SF. When
                given, returns the ``(min, max)`` St range that keeps the
                material's safety factor between ``safety_factor`` and
                ``max_safety_factor`` — useful to avoid picking a material
                so strong it is needlessly over-designed.

        Returns:
            float: Required St in MPa, when ``max_safety_factor`` is
            omitted.
            tuple: ``(min_St, max_St)`` in MPa, when ``max_safety_factor``
            is given.

        Raises:
            ValueError: If ``safety_factor`` is not strictly positive, or
                ``max_safety_factor`` is given but not strictly greater
                than ``safety_factor``.
        """
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        sigma = self.bending_stress_pinion
        if self.bending_stress_gear is not None:
            sigma = max(sigma, self.bending_stress_gear)
        unit_St = sigma * self.temperature_factor * self.YZ / self.YN
        min_St = safety_factor * unit_St
        if max_safety_factor is None:
            return min_St
        if max_safety_factor <= safety_factor:
            raise ValueError(
                "max_safety_factor must be strictly greater than safety_factor"
            )
        return min_St, max_safety_factor * unit_St

    def required_Sc(self, safety_factor=1.0, max_safety_factor=None):
        """
        Allowable contact stress number a material must have.

        Inverts ``SH = Sc ZN ZW / (Y_theta YZ sigma_H)`` for Sc. ZW is 1
        unless both hardnesses are known, which is the conservative
        assumption while selecting a material.

        Args:
            safety_factor (float): Target (minimum) pitting safety factor
                SH (default 1.0).
            max_safety_factor (float): Optional upper bound on SH. When
                given, returns the ``(min, max)`` Sc range that keeps the
                material's safety factor between ``safety_factor`` and
                ``max_safety_factor``.

        Returns:
            float: Required Sc in MPa, when ``max_safety_factor`` is
            omitted.
            tuple: ``(min_Sc, max_Sc)`` in MPa, when ``max_safety_factor``
            is given.

        Raises:
            ValueError: If ``safety_factor`` is not strictly positive, or
                ``max_safety_factor`` is given but not strictly greater
                than ``safety_factor``.
        """
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        unit_Sc = (self.contact_stress * self.temperature_factor
                   * self.YZ / (self.ZN * self.ZW))
        min_Sc = safety_factor * unit_Sc
        if max_safety_factor is None:
            return min_Sc
        if max_safety_factor <= safety_factor:
            raise ValueError(
                "max_safety_factor must be strictly greater than safety_factor"
            )
        return min_Sc, max_safety_factor * unit_Sc

    def required_strengths(self, safety_factor=1.0, max_safety_factor=None):
        """
        Required St and Sc in one call.

        Convenience wrapper around :meth:`required_St` and
        :meth:`required_Sc` for picking a single material that satisfies
        both the bending and pitting sides at once.

        Args:
            safety_factor (float): Target (minimum) safety factor,
                forwarded to both (default 1.0).
            max_safety_factor (float): Optional upper bound, forwarded to
                both to also get a ``(min, max)`` range for each.

        Returns:
            dict: ``{"St": ..., "Sc": ...}`` in MPa. Each value is a
            float when ``max_safety_factor`` is omitted, or a
            ``(min, max)`` tuple when it is given.

        Raises:
            ValueError: Same conditions as :meth:`required_St` /
                :meth:`required_Sc`.
        """
        return {
            "St": self.required_St(safety_factor, max_safety_factor),
            "Sc": self.required_Sc(safety_factor, max_safety_factor),
        }

    def __repr__(self):
        if not self.has_allowables:
            return (
                f"AGMARating(sigma_F={self.bending_stress_pinion:.1f} MPa, "
                f"sigma_H={self.contact_stress:.1f} MPa, no material)"
            )
        return (
            f"AGMARating(SF_pinion={self.SF_pinion:.2f}, "
            f"SH={self.SH:.2f})"
        )
