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

    Args:
        pinion (CylindricalGear): The pinion.
        gear (CylindricalGear): The mating gear; ``None`` for a pinion
            driving a rack.

    Returns:
        float: Geometry factor I (ZI, dimensionless).

    Raises:
        ValueError: If contact falls below the pinion base circle
            (rho_p <= 0) or the mesh has tip interference (rho_g <= 0).
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
        inv_rho_g = 0.0
    else:
        c = pinion.working_center_distance_with(gear)
        rho_g = c * math.sin(phi_t) - rho_p
        if rho_g <= 0:
            raise ValueError("Gear flank curvature is non-positive "
                             "(rho_g <= 0); mesh has tip interference")
        inv_rho_g = 1.0 / rho_g
    return (math.cos(phi_t)
            / ((1.0 / rho_p + inv_rho_g) * pinion.pitch_diameter))


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

    Evaluates all modification factors and stresses eagerly at
    construction; results are plain attributes. The first gear is the
    pinion (driver); the mate may be a gear or a :class:`Rack`.

    Note: the digitized J-factor and Lewis tables assume standard
    (profile shift x = 0) tooth proportions; ratings for
    profile-shifted gears are approximate.

    Attributes:
        Ft (float): Tangential force in N.
        pitch_line_velocity (float): In m/s.
        Kv, KH, KB, Ks, Ko (float): Modification factors.
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
                 tangential_force=None, Ko=1.0, Qv=7, Ks=1.0, ZR=1.0,
                 KB=1.0, life_cycles=1e7, reliability=0.99, grade=1,
                 hardness_HB=None, gear_hardness_HB=None, St=None, Sc=None,
                 YJ_pinion=None, YJ_gear=None, condition="commercial",
                 crowned=False, temperature_factor=1.0, temperature_celsius=60):
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
            Ks (float): Size factor (default 1.0).
            ZR (float): Surface-condition factor (default 1.0).
            KB (float): Rim-thickness factor (default 1.0, solid gear);
                see :func:`rim_thickness_factor`.
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
                material allowable can be determined, or both (or neither)
                temperature inputs are given.
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
        # Fall back to a GearMaterial on the pinion for the hardness data.
        if hardness_HB is None and isinstance(pinion.material, GearMaterial):
            hardness_HB = pinion.material.hardness_HB
            grade = pinion.material.grade
        if hardness_HB is None and (St is None or Sc is None):
            raise ValueError(
                "Give 'hardness_HB' (through-hardened steel fits) or "
                "explicit 'St' and 'Sc' allowable stress numbers in MPa"
            )
        if (temperature_factor != 1.0) and (temperature_celsius is not None):
            raise ValueError(
                "Give 'temperature_factor' (explicit Y_theta) or "
                "'temperature_celsius' (oil temperature), not both"
            )
        if temperature_celsius is not None:
            temperature_factor = _temperature_factor_from_temp(temperature_celsius)

        self.pinion = pinion
        self.gear = gear
        self.Ko = Ko
        self.Ks = Ks
        self.ZR = ZR
        self.KB = KB
        self.temperature_factor = temperature_factor

        is_rack = isinstance(gear, Rack)
        mating_teeth = "rack" if is_rack else gear.teeth
        # A rack is kinematically an infinite gear: mG -> infinity;
        # use a large finite ratio for the I factor.
        self.gear_ratio = 1e6 if is_rack else gear.teeth / pinion.teeth
        if self.gear_ratio < 1:
            raise ValueError(
                "The pinion must be the smaller member (gear ratio >= 1)"
            )

        # Effective face width: the narrower of the two members.
        b = pinion.face_width
        if not is_rack and gear.face_width is not None:
            b = min(b, gear.face_width)
        self.face_width = b

        # Load and velocity.
        self.pitch_line_velocity = pinion.pitch_line_velocity(pinion_speed_rpm)
        if tangential_force is not None:
            if tangential_force <= 0:
                raise ValueError("Tangential force must be strictly positive")
            self.Ft = tangential_force
        else:
            self.Ft = pinion.tangential_force(power_kw, pinion_speed_rpm)

        # Modification factors.
        self.Kv = dynamic_factor(self.pitch_line_velocity, Qv)
        self.KH = load_distribution_factor(b, pinion.pitch_diameter,
                                           condition=condition,
                                           crowned=crowned)
        self.YJ_pinion = (YJ_pinion if YJ_pinion is not None else
                          agma_data.geometry_factor_J(
                              pinion.teeth, mating_teeth,
                              pinion.helix_angle))
        if is_rack:
            self.YJ_gear = None
        else:
            self.YJ_gear = (YJ_gear if YJ_gear is not None else
                            agma_data.geometry_factor_J(
                                gear.teeth, pinion.teeth,
                                gear.helix_angle))
        self.ZI = geometry_factor_I(pinion, gear=None if is_rack else gear)
        gear_props = (pinion.material_properties if is_rack
                      else gear.material_properties)
        self.ZE = elastic_coefficient(pinion.material_properties, gear_props)

        # Life, reliability and hardness factors.
        self.YN = bending_life_factor(life_cycles)
        self.ZN = contact_life_factor(life_cycles)
        self.YZ = agma_data.reliability_factor(reliability)
        # Fall back to a GearMaterial on the gear for its hardness (ZW).
        if (gear_hardness_HB is None and not is_rack
                and isinstance(gear.material, GearMaterial)):
            gear_hardness_HB = gear.material.hardness_HB
        if gear_hardness_HB is not None and hardness_HB is not None:
            self.ZW = hardness_ratio_factor(hardness_HB, gear_hardness_HB,
                                            self.gear_ratio)
        else:
            self.ZW = 1.0

        # Allowable stress numbers.
        self.St = (St if St is not None else
                   agma_data.allowable_bending_stress(hardness_HB, grade))
        self.Sc = (Sc if Sc is not None else
                   agma_data.allowable_contact_stress(hardness_HB, grade))

        # ------------------------------------------------------------
        # Stresses (metric AGMA 2101 fundamental equations)
        # ------------------------------------------------------------
        mt = pinion.transverse_module
        common = self.Ft * self.Ko * self.Kv * self.Ks
        self.bending_stress_pinion = (common / (b * mt)
                                      * self.KH * self.KB / self.YJ_pinion)
        if self.YJ_gear is not None:
            self.bending_stress_gear = (common / (b * mt)
                                        * self.KH * self.KB / self.YJ_gear)
        else:
            self.bending_stress_gear = None
        self.contact_stress = self.ZE * math.sqrt(
            common * self.KH / (pinion.pitch_diameter * b)
            * self.ZR / self.ZI
        )

        # Allowables and safety factors.
        self.allowable_bending_stress = (
            self.St * self.YN / (self.temperature_factor * self.YZ)
        )
        self.allowable_contact_stress = (
            self.Sc * self.ZN * self.ZW / (self.temperature_factor * self.YZ)
        )
        self.SF_pinion = (self.allowable_bending_stress
                          / self.bending_stress_pinion)
        self.SF_gear = (self.allowable_bending_stress
                        / self.bending_stress_gear
                        if self.bending_stress_gear else None)
        self.SH = self.allowable_contact_stress / self.contact_stress

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
            f"KH={self.KH:.3f} KB={self.KB:.2f}",
            f"Geometry: YJ_pinion={self.YJ_pinion:.3f} "
            + (f"YJ_gear={self.YJ_gear:.3f} " if self.YJ_gear else "")
            + f"ZI={self.ZI:.4f} ZE={self.ZE:.1f} sqrt(MPa)",
            f"Bending stress (pinion): {self.bending_stress_pinion:.1f} MPa",
        ]
        if self.bending_stress_gear is not None:
            lines.append(
                f"Bending stress (gear):   {self.bending_stress_gear:.1f} MPa"
            )
        lines += [
            f"Contact stress:          {self.contact_stress:.1f} MPa",
            f"Allowable bending:       {self.allowable_bending_stress:.1f} MPa",
            f"Allowable contact:       {self.allowable_contact_stress:.1f} MPa",
            f"Safety factors: SF_pinion={self.SF_pinion:.2f} "
            + (f"SF_gear={self.SF_gear:.2f} " if self.SF_gear else "")
            + f"SH={self.SH:.2f}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"AGMARating(SF_pinion={self.SF_pinion:.2f}, "
            f"SH={self.SH:.2f})"
        )
