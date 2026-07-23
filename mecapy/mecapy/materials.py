"""Material database, properties and fatigue analysis for MecaPy.

Units convention: the underlying database (``constants.MATERIALS``) is strict
SI (Pa, kg/m^3, J/(kg*K)). The :class:`Material` class API works entirely in
**MPa** for strengths, **mm** for the size (diameter) input and **degrees
Celsius** for temperature, converting from the Pa database at the boundary.
Fatigue relations follow Shigley's Mechanical Engineering Design, Ch. 6.
"""

import math
from statistics import NormalDist

from .utils import constants


def get_material_properties(material_name):
    """
    Get properties for a specific material.

    Args:
        material_name (str): Name of the material

    Returns:
        dict: Dictionary of material properties

    Raises:
        ValueError: If material is not found in database
    """
    if material_name not in constants.MATERIALS:
        raise ValueError(f"Material '{material_name}' not found in database")
    return constants.MATERIALS[material_name]


def get_available_materials():
    """Get list of available materials in database."""
    return list(constants.MATERIALS.keys())


def add_custom_material(name, properties):
    """
    Add a custom material to the database.

    Args:
        name (str): Material name
        properties (dict): Dictionary of material properties
    """
    constants.MATERIALS[name] = properties


# Marin surface-condition factor ka = a * Sut**b with Sut in MPa
# (Shigley Table 6-2, MPa column).
SURFACE_FINISH_FACTORS = {
    "ground": (1.58, -0.085),
    "machined": (4.51, -0.265),
    "cold_drawn": (4.51, -0.265),
    "hot_rolled": (57.7, -0.718),
    "as_forged": (272.0, -0.995),
}

# Marin load factor kc (Shigley Eq. 6-26).
LOAD_FACTORS = {"bending": 1.0, "axial": 0.85, "torsion": 0.59, "combined": 1}

# Marin temperature factor kd = ST/SRT vs temperature in deg C
# (Shigley Table 6-4); linearly interpolated between table points.
TEMPERATURE_FACTORS = {
    20: 1.000, 50: 1.010, 100: 1.020, 150: 1.025, 200: 1.020,
    250: 1.000, 300: 0.975, 350: 0.943, 400: 0.900, 450: 0.843,
    500: 0.768, 550: 0.672, 600: 0.549,
}

# Marin reliability factor ke vs survival probability (Shigley Table 6-5).
RELIABILITY_FACTORS = {
    0.5: 1.000, 0.90: 0.897, 0.95: 0.868, 0.99: 0.814,
    0.999: 0.753, 0.9999: 0.702, 0.99999: 0.659,
}

# Mean-stress fatigue criteria (Shigley Ch. 6). Goodman/Gerber use Sut as
# the mean-axis intercept; Soderberg/ASME-elliptic use Sy.
_FATIGUE_CRITERIA = ("goodman", "soderberg", "gerber", "asme_elliptic")

# Sut below which the fatigue-strength fraction f is taken as 0.9
# (70 kpsi, Shigley Fig. 6-18).
_F_FRACTION_SUT_MIN = 482.6  # MPa

# High-cycle fatigue life bounds for the S-N (Woehler) line (Shigley
# Eq. 6-13): Sf = a*N^b holds from 10^3 cycles to the 10^6-cycle endurance
# knee; below 10^3 is low-cycle fatigue, above 10^6 the strength is flat Se.
_SN_LOW_CYCLE = 1e3
_SN_KNEE_CYCLES = 1e6
_MPA_PER_KPSI = constants.PSI_TO_MPA * 1e3

# Validity range of the rotating-size factor kb (Shigley Eq. 6-20), mm.
_KB_DIAMETER_MIN = 2.79
_KB_DIAMETER_MAX = 254.0

# 95%-stress-area coefficient for the equivalent diameter (Shigley Eq. 6-25):
# a rotating round section has A_0.95sigma = 0.0766 * d^2.
_A95_COEFF = 0.0766

# Equivalent-diameter factor for a NON-rotating solid round section
# (Table 6-3: A_0.95sigma = 0.01046 d^2 -> d_e = sqrt(0.01046/0.0766) d).
_NONROTATING_ROUND_FACTOR = 0.370

# Neuber constant sqrt(a) curve-fit for steels (Shigley Eqs. 6-35a/6-35b):
# sqrt(a) [in^0.5] = c0 + c1*Sut + c2*Sut^2 + c3*Sut^3 with Sut in kpsi.
# Bending and axial share one fit; torsion (shear) uses the other. Axial
# uses the bending/axial curve. Combined loading has no single notch curve.
_NEUBER_COEFFS = {
    "bending": (0.246, -3.08e-3, 1.51e-5, -2.67e-8),
    "axial": (0.246, -3.08e-3, 1.51e-5, -2.67e-8),
    "torsion": (0.190, -2.51e-3, 1.35e-5, -2.67e-8),
}

_LOAD_CASE_KEYS = ("stress_amplitude", "mean_stress", "cycles", "label")


class Material:
    """
    Material with static strength data and Shigley Ch. 6 fatigue analysis.

    Strengths default from the material database (converted Pa -> MPa) and
    can be overridden explicitly. The endurance limit follows the Marin
    equation Se = ka*kb*kc*ke*kf*Se' with Se' estimated from the
    temperature-corrected ultimate strength Sut_T = kd*Sut (steel
    correlation); temperature enters through Sut_T (which also drives ka,
    f, sn_a, sn_b), not as a separate kd factor on Se. The S-N (Woehler)
    and Goodman diagrams can be plotted, optionally annotated with a
    component's cyclic load cases.

    ``Material`` is stateless with respect to loading: it holds no load
    cases. A component's cyclic spectrum and Miner accumulation live on the
    component itself (see :class:`CyclicLoadCases`), so one ``Material``
    instance can be shared across components without cross-contamination.

    Every Marin input (surface finish, loading, diameter, temperature,
    reliability, kf) and both strengths are settable; all derived quantities
    (ka..ke, Se, S-N coefficients, damage) recompute automatically.

    Attributes:
        base_material (str): Database material this instance is based on
            (fixed at construction; supplies density, moduli, etc.).
        ultimate_strength (float): Ultimate tensile strength Sut in MPa
            (room temperature).
        ultimate_strength_temperature (float): Temperature-corrected
            ultimate strength Sut_T = kd*Sut in MPa (derived, read-only).
            Feeds ka, se_prime, f, sn_a and sn_b.
        yield_strength (float): Yield strength Sy in MPa.
        surface_finish (str): Surface condition for ka (key of
            ``SURFACE_FINISH_FACTORS``).
        loading (str): Load type for kb/kc ("bending", "axial", "torsion").
        diameter (float or None): Section diameter in mm for kb
            (None -> kb = 1).
        rotating (bool): Whether the round section rotates; if False kb
            uses the equivalent diameter 0.370*diameter (Table 6-3).
        temperature (float): Operating temperature in deg C for kd.
        reliability (float): Survival probability for ke, in [0.5, 1).
            Tabulated values (``RELIABILITY_FACTORS``) use Table 6-5;
            any other value is computed from Shigley Eq. 6-30.
        kf (float): Marin miscellaneous-effects factor (default: 1.0).
            Lumps stress concentration, corrosion, plating, residual
            stress and other effects not covered by ka..ke. Must be
            strictly positive.
        name (str): Optional identifier.

    Notch sensitivity q (``notch_sensitivity``) and the fatigue
    stress-concentration factor Kf (``fatigue_stress_concentration_factor``)
    are pure functions of their arguments: they read the material's strength
    but store nothing, so one ``Material`` instance can be shared by many
    components without cross-contamination.
    """

    def __init__(self, base_material="steel", ultimate_strength=None,
                 yield_strength=None, surface_finish="machined",
                 loading="bending", diameter=None, rotating=True,
                 temperature=20.0, reliability=0.5, kf=1.0, name=None):
        """
        Initialize a Material.

        Args:
            base_material (str): Database material supplying defaults and
                the non-strength properties (default: "steel"). Must exist
                in the database.
            ultimate_strength (float): Sut in MPa. Defaults from the
                database row. Must be strictly positive.
            yield_strength (float): Sy in MPa. Defaults from the database
                row. Must be strictly positive and below Sut.
            surface_finish (str): Surface condition for ka (default:
                "machined"). One of ``SURFACE_FINISH_FACTORS``.
            loading (str): Load type for kc (default: "bending"). One of
                ``LOAD_FACTORS``.
            diameter (float): Optional section diameter in mm for the size
                factor kb. Must be within [2.79, 254] mm (Shigley Eq. 6-20
                validity range). None gives kb = 1.
            rotating (bool): Whether the round section rotates (default:
                True). If False, kb uses the non-rotating equivalent
                diameter d_e = 0.370*diameter (Table 6-3).
            temperature (float): Operating temperature in deg C (default:
                20). Must be within [20, 600] (Table 6-4 range).
            reliability (float): Survival probability (default: 0.5, i.e.
                ke = 1). Must be in [0.5, 1); tabulated values use Table
                6-5, any other value is computed from Shigley Eq. 6-30.
            kf (float): Marin miscellaneous-effects factor (default: 1.0,
                i.e. no effect). Must be strictly positive.
            name (str): Optional identifier.

        Raises:
            ValueError: If ``base_material`` is unknown, if a strength is
                needed but neither given nor present in the database row,
                or if any input fails its validation above.
        """
        row = get_material_properties(base_material)
        self.base_material = base_material
        self.name = name

        if ultimate_strength is None:
            if "ultimate_strength" not in row:
                raise ValueError(
                    f"Material '{base_material}' has no 'ultimate_strength' in the "
                    "database; pass ultimate_strength= explicitly (MPa)"
                )
            ultimate_strength = row["ultimate_strength"] / 1e6
        if yield_strength is None:
            if "yield_strength" not in row:
                raise ValueError(
                    f"Material '{base_material}' has no 'yield_strength' in the "
                    "database; pass yield_strength= explicitly (MPa)"
                )
            yield_strength = row["yield_strength"] / 1e6

        # Set Sut before Sy: the Sy setter checks Sy < Sut.
        self.ultimate_strength = ultimate_strength
        self.yield_strength = yield_strength
        self.surface_finish = surface_finish
        self.loading = loading
        self.diameter = diameter
        self.rotating = rotating
        self.temperature = temperature
        self.reliability = reliability
        self.kf = kf

    # ---- Settable primary inputs ----

    @property
    def ultimate_strength(self):
        """float: Ultimate tensile strength Sut in MPa."""
        return self._ultimate_strength

    @ultimate_strength.setter
    def ultimate_strength(self, value):
        if value <= 0:
            raise ValueError("Ultimate strength must be strictly positive")
        self._ultimate_strength = value

    @property
    def yield_strength(self):
        """float: Yield strength Sy in MPa."""
        return self._yield_strength

    @yield_strength.setter
    def yield_strength(self, value):
        if value <= 0:
            raise ValueError("Yield strength must be strictly positive")
        if value >= self.ultimate_strength:
            raise ValueError(
                "Yield strength must be below the ultimate strength "
                f"({self.ultimate_strength} MPa)"
            )
        self._yield_strength = value

    @property
    def surface_finish(self):
        """str: Surface condition for the Marin factor ka."""
        return self._surface_finish

    @surface_finish.setter
    def surface_finish(self, value):
        if value not in SURFACE_FINISH_FACTORS:
            available = ", ".join(sorted(SURFACE_FINISH_FACTORS))
            raise ValueError(
                f"Unknown surface finish {value!r}. Available: {available}"
            )
        self._surface_finish = value

    @property
    def loading(self):
        """str: Load type for the Marin factors kb and kc."""
        return self._loading

    @loading.setter
    def loading(self, value):
        if value not in LOAD_FACTORS:
            available = ", ".join(sorted(LOAD_FACTORS))
            raise ValueError(f"Unknown loading {value!r}. Available: {available}")
        self._loading = value

    @property
    def diameter(self):
        """float or None: Rotating-section diameter in mm for kb."""
        return self._diameter

    @diameter.setter
    def diameter(self, value):
        if value is not None and not (
            _KB_DIAMETER_MIN <= value <= _KB_DIAMETER_MAX
        ):
            raise ValueError(
                "Diameter must be within "
                f"[{_KB_DIAMETER_MIN}, {_KB_DIAMETER_MAX}] mm (Eq. 6-20 "
                "validity range) or None for kb = 1"
            )
        self._diameter = value

    @property
    def rotating(self):
        """bool: Whether the round section rotates (affects kb)."""
        return self._rotating

    @rotating.setter
    def rotating(self, value):
        if not isinstance(value, bool):
            raise ValueError("rotating must be a bool")
        self._rotating = value

    @property
    def effective_diameter(self):
        """float or None: Diameter kb uses, in mm.

        The physical ``diameter`` when rotating, else the non-rotating
        equivalent d_e = 0.370*diameter (Table 6-3). None when no diameter
        is set.
        """
        if self.diameter is None:
            return None
        if self.rotating:
            return self.diameter
        return _NONROTATING_ROUND_FACTOR * self.diameter

    def eq_diameter(self, A95):
        """
        Set the size-factor diameter from a 95%-stress area (Eq. 6-25).

        For a non-round or non-rotating section the size factor kb uses an
        equivalent rotating diameter d_e = sqrt(A_0.95sigma / 0.0766),
        where A_0.95sigma is the area stressed to at least 95% of the
        maximum. This sets ``self.diameter`` to d_e (so kb picks it up) and
        returns d_e.

        Args:
            A95 (float): 95%-stress area A_0.95sigma in mm^2. Must be
                strictly positive.

        Returns:
            float: Equivalent diameter d_e in mm.

        Raises:
            ValueError: If ``A95`` is not strictly positive, or the
                resulting d_e falls outside the kb validity range
                [2.79, 254] mm.
        """
        if A95 <= 0:
            raise ValueError("A95 must be strictly positive")
        d_e = math.sqrt(A95 / _A95_COEFF)
        self.diameter = d_e  # revalidates against the kb range
        return d_e

    @property
    def temperature(self):
        """float: Operating temperature in deg C for kd."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        points = sorted(TEMPERATURE_FACTORS)
        if not points[0] <= value <= points[-1]:
            raise ValueError(
                f"Temperature must be within [{points[0]}, {points[-1]}] deg C "
                "(Table 6-4 range)"
            )
        self._temperature = value

    @property
    def reliability(self):
        """float: Survival probability for ke."""
        return self._reliability

    @reliability.setter
    def reliability(self, value):
        if not 0.5 <= value < 1:
            raise ValueError(
                f"Reliability {value!r} must be in [0.5, 1)"
            )
        self._reliability = value

    # ---- Marin factors and endurance limit (recomputed on access) ----

    @property
    def ka(self):
        """float: Marin surface-condition factor, a*Sut_T^b (Table 6-2).

        Evaluated at the temperature-corrected ultimate strength
        ``ultimate_strength_temperature`` (Sut_T = kd*Sut), so temperature
        enters the endurance limit here instead of as a separate kd factor.
        """
        a, b = SURFACE_FINISH_FACTORS[self.surface_finish]
        return a * self.ultimate_strength_temperature ** b

    @property
    def kb(self):
        """float: Marin size factor (Eq. 6-19/6-20).

        1.0 for axial loading or when no diameter is set; otherwise
        (de/7.62)^-0.107 for de <= 51 mm and 1.51*de^-0.157 above, where
        de is ``effective_diameter`` (the physical diameter when rotating,
        else the non-rotating equivalent 0.370*diameter).
        """
        de = self.effective_diameter
        if self.loading == "axial" or de is None:
            return 1.0
        if de <= 51:
            return (de / 7.62) ** -0.107
        return 1.51 * de ** -0.157

    @property
    def kc(self):
        """float: Marin load factor (Eq. 6-26)."""
        return LOAD_FACTORS[self.loading]

    @property
    def kd(self):
        """float: Marin temperature factor, interpolated from Table 6-4."""
        points = sorted(TEMPERATURE_FACTORS)
        t = self.temperature
        for lo, hi in zip(points, points[1:]):
            if t <= hi:
                f_lo, f_hi = TEMPERATURE_FACTORS[lo], TEMPERATURE_FACTORS[hi]
                return f_lo + (f_hi - f_lo) * (t - lo) / (hi - lo)
        return TEMPERATURE_FACTORS[points[-1]]

    @property
    def ultimate_strength_temperature(self):
        """float: Temperature-corrected ultimate strength Sut_T in MPa.

        Sut_T = kd*Sut (Shigley Sec. 6-9, kd = ST/SRT). All fatigue-strength
        quantities derived from Sut — ``ka``, ``se_prime``, ``f``, ``sn_a``
        and ``sn_b`` — use this value so temperature is applied once, at the
        strength, rather than as a separate kd multiplier on Se. Equals Sut
        at room temperature (kd = 1).
        """
        return self.kd * self.ultimate_strength

    @property
    def ke(self):
        """float: Marin reliability factor (Shigley Table 6-5 / Eq. 6-30).

        Uses the tabulated value for a listed reliability, otherwise
        computes ke = 1 - 0.08*za with za = Phi^-1(R) (the formula the
        table is built from).
        """
        if self.reliability in RELIABILITY_FACTORS:
            return RELIABILITY_FACTORS[self.reliability]
        return 1 - 0.08 * NormalDist().inv_cdf(self.reliability)

    @property
    def kf(self):
        """float: Marin miscellaneous-effects factor."""
        return self._kf

    @kf.setter
    def kf(self, value):
        if value <= 0:
            raise ValueError("kf must be strictly positive")
        self._kf = value

    @property
    def se_prime(self):
        """float: Unmodified endurance limit Se' in MPa.

        Steel correlation (Eq. 6-8): 0.5*Sut_T for Sut_T <= 1400 MPa, else
        700 MPa, evaluated at the temperature-corrected ultimate strength
        ``ultimate_strength_temperature``. For non-ferrous materials this is
        an approximation only — many (e.g. aluminum) have no true endurance
        limit.
        """
        sut_t = self.ultimate_strength_temperature
        if sut_t <= 1400:
            return 0.5 * sut_t
        return 700.0

    @property
    def endurance_limit(self):
        """float: Corrected endurance limit Se in MPa.

        Marin equation (Eq. 6-17): Se = ka*kb*kc*ke*kf*Se', where kf is the
        miscellaneous-effects factor (default 1.0). The temperature factor
        kd is NOT a separate multiplier here — temperature is applied at the
        strength via ``ultimate_strength_temperature`` (Sut_T = kd*Sut),
        which drives ka and Se'. Applying kd again would double-count it.
        """
        return (self.ka * self.kb * self.kc * self.ke * self.kf
                * self.se_prime)

    # ---- Notch sensitivity (Neuber) ----

    def _neuber_sqrt_a(self, loading):
        """float: Neuber constant sqrt(a) in sqrt(mm) for a loading type.

        Steel curve fit (Shigley Eqs. 6-35a/6-35b) evaluated in US units
        (sqrt(in), Sut in kpsi), then converted to sqrt(mm) at the
        boundary. Raises ValueError for a loading without a notch curve
        (e.g. "combined").
        """
        if loading not in _NEUBER_COEFFS:
            available = ", ".join(sorted(_NEUBER_COEFFS))
            raise ValueError(
                f"Notch sensitivity is defined for {available} loading, "
                f"not {loading!r}"
            )
        c0, c1, c2, c3 = _NEUBER_COEFFS[loading]
        sut = self.ultimate_strength / _MPA_PER_KPSI  # kpsi
        sqrt_a_in = c0 + c1 * sut + c2 * sut ** 2 + c3 * sut ** 3
        return sqrt_a_in * math.sqrt(constants.IN_TO_MM)

    @property
    def neuber_constant(self):
        """float: Neuber constant sqrt(a) in sqrt(mm) for ``self.loading``.

        Steel correlation (Shigley Eq. 6-35), bending/axial vs. torsion
        selected from ``self.loading``. Raises ValueError if the current
        loading has no notch curve (e.g. "combined").
        """
        return self._neuber_sqrt_a(self.loading)

    def _q_neuber(self, loading, radius):
        """float: Neuber notch sensitivity q = 1/(1 + sqrt(a)/sqrt(r)), pure."""
        sqrt_a = self._neuber_sqrt_a(loading)
        return 1.0 / (1.0 + sqrt_a / math.sqrt(radius))

    def notch_sensitivity(self, radius, loading=None):
        """
        Notch sensitivity q for a notch radius (Neuber, Shigley Eq. 6-34).

        Pure function: q = 1 / (1 + sqrt(a)/sqrt(r)), with sqrt(a) the Neuber
        constant (steel curve fit, Eq. 6-35) and r the notch root radius. It
        reads the material's strength but stores nothing, so calling it any
        number of times leaves the ``Material`` unchanged. For
        ``loading="combined"`` both curves are evaluated and returned as a
        pair. q feeds ``fatigue_stress_concentration_factor``
        (Kf = 1 + q*(Kt - 1)).

        Args:
            radius (float): Notch root radius r in mm. Must be strictly
                positive.
            loading (str): Load type selecting the Neuber curve (default:
                ``self.loading``). One of "bending", "axial", "torsion",
                "combined"; bending and axial share the same curve.

        Returns:
            float: Notch sensitivity q in [0, 1] for a single loading, or a
            ``(q, q_shear)`` tuple of two such values for "combined".

        Raises:
            ValueError: If ``radius`` is not strictly positive, or
                ``loading`` has no notch curve (an unknown loading raises;
                "combined" is handled).
        """
        if radius <= 0:
            raise ValueError("Notch radius must be strictly positive")
        loading = self.loading if loading is None else loading
        if loading == "combined":
            return (self._q_neuber("bending", radius),
                    self._q_neuber("torsion", radius))
        return self._q_neuber(loading, radius)

    @staticmethod
    def fatigue_stress_concentration_factor(kt, q=1.0):
        """
        Fatigue stress-concentration factor Kf from Kt and sensitivity q.

        Kf = 1 + q*(Kt - 1) (Shigley Eq. 6-32). q = 1.0 (the default) is the
        fully notch-sensitive, conservative case Kf = Kt; q = 0 gives Kf = 1
        (no notch effect). Get a q value from ``notch_sensitivity``.

        Args:
            kt (float): Theoretical (geometric) stress-concentration factor
                Kt. Must be >= 1.
            q (float): Notch sensitivity (default: 1.0). Within [0, 1].

        Returns:
            float: Fatigue stress-concentration factor Kf (>= 1).

        Raises:
            ValueError: If ``kt`` < 1 or ``q`` is outside [0, 1].
        """
        if kt < 1:
            raise ValueError("Kt (stress-concentration factor) must be >= 1")
        if not 0 <= q <= 1:
            raise ValueError("q (notch sensitivity) must be within [0, 1]")
        return 1.0 + q * (kt - 1.0)

    # ---- S-N (Woehler) curve ----

    @property
    def f(self):
        """float: Fatigue-strength fraction of Sut_T at 10^3 cycles.

        Fig. 6-18 fit: 0.9 below 482.6 MPa (70 kpsi), else the quadratic
        1.06 - 2.8e-3*Sut_T + 6.9e-6*Sut_T^2 with Sut_T in kpsi. Uses the
        temperature-corrected ``ultimate_strength_temperature``.
        """
        sut_t = self.ultimate_strength_temperature
        if sut_t < _F_FRACTION_SUT_MIN:
            return 0.9
        sut_kpsi = sut_t / _MPA_PER_KPSI
        return 1.06 - 2.8e-3 * sut_kpsi + 6.9e-6 * sut_kpsi ** 2

    @property
    def sn_a(self):
        """float: S-N curve coefficient a in MPa, (f*Sut_T)^2/Se (Eq. 6-14).

        Uses the temperature-corrected ``ultimate_strength_temperature``.
        """
        return ((self.f * self.ultimate_strength_temperature) ** 2
                / self.endurance_limit)

    @property
    def sn_b(self):
        """float: S-N curve exponent b, -(1/3)*log10(f*Sut_T/Se) (Eq. 6-15).

        Uses the temperature-corrected ``ultimate_strength_temperature``.
        """
        return -math.log10(self.f * self.ultimate_strength_temperature
                           / self.endurance_limit) / 3

    def fatigue_strength(self, cycles):
        """
        Fatigue strength Sf at a given life N from the S-N curve (Eq. 6-13).

        Completely-reversed high-cycle fatigue: Sf = a*N^b (a = ``sn_a``,
        b = ``sn_b``) for 10^3 <= N <= 10^6, flattening to the endurance
        limit ``Se`` for N >= 10^6. This is the inverse of
        ``cycles_to_failure`` on the reversed S-N line: Sf at
        ``cycles_to_failure(sigma)`` recovers sigma.

        Args:
            cycles (float): Life N in cycles. Must be >= 10^3 (the
                high-cycle fatigue range; N < 10^3 is low-cycle fatigue,
                not modeled by this line).

        Returns:
            float: Fully reversed fatigue strength Sf in MPa; the endurance
            limit Se for N >= 10^6.

        Raises:
            ValueError: If ``cycles`` < 10^3.
        """
        if cycles < _SN_LOW_CYCLE:
            raise ValueError(
                f"Cycles must be >= {_SN_LOW_CYCLE:g} (high-cycle fatigue "
                "range); low-cycle fatigue (N < 10^3) is not modeled"
            )
        if cycles >= _SN_KNEE_CYCLES:
            return self.endurance_limit
        return self.sn_a * cycles ** self.sn_b

    def _equivalent_reversed_stress(self, stress_amplitude, mean_stress,
                                    criterion):
        """float: Completely-reversed stress sigma_ar for a criterion, MPa.

        Folds the mean stress into an equivalent fully reversed amplitude
        (Shigley Table 6-7). Callers must have validated the (sigma_a,
        sigma_m, criterion) triple first, so the denominators stay positive.
        """
        s = self._mean_strength(criterion)
        if criterion == "gerber":
            return stress_amplitude / (1 - (mean_stress / s) ** 2)
        if criterion == "asme_elliptic":
            return stress_amplitude / math.sqrt(1 - (mean_stress / s) ** 2)
        return stress_amplitude / (1 - mean_stress / s)  # goodman, soderberg

    def _fatigue_nf(self, stress_amplitude, mean_stress, criterion):
        """float: Infinite-life fatigue safety factor for a criterion.

        Closed forms from Shigley Eqs. 6-45..6-50. Callers validate first.
        """
        se, sut, sy = (self.endurance_limit, self.ultimate_strength,
                       self.yield_strength)
        if criterion == "soderberg":
            return 1 / (stress_amplitude / se + mean_stress / sy)
        if criterion == "asme_elliptic":
            return 1 / math.sqrt((stress_amplitude / se) ** 2
                                 + (mean_stress / sy) ** 2)
        if criterion == "gerber":
            if mean_stress == 0:
                return se / stress_amplitude
            return (0.5 * (sut / mean_stress) ** 2 * (stress_amplitude / se)
                    * (-1 + math.sqrt(1 + (2 * mean_stress * se
                                           / (sut * stress_amplitude)) ** 2)))
        return 1 / (stress_amplitude / se + mean_stress / sut)  # goodman

    def cycles_to_failure(self, stress_amplitude, mean_stress=0.0,
                          criterion="goodman"):
        """
        Cycles to failure N for a cyclic stress (mean-stress fold + S-N).

        The mean stress is folded into an equivalent fully reversed
        amplitude sigma_ar with the chosen criterion (Shigley Table 6-7),
        then N = (sigma_ar/a)^(1/b) from the S-N curve (Eq. 6-16).

        Args:
            stress_amplitude (float): Alternating stress sigma_a in MPa.
                Must be strictly positive.
            mean_stress (float): Mean stress sigma_m in MPa (default: 0).
                Must be within [0, Sut) for goodman/gerber or [0, Sy) for
                soderberg/asme_elliptic.
            criterion (str): Mean-stress criterion (default: "goodman").
                One of ``_FATIGUE_CRITERIA``.

        Returns:
            float: Cycles to failure; ``math.inf`` if the equivalent
            amplitude is at or below the endurance limit.

        Raises:
            ValueError: If ``stress_amplitude`` is not strictly positive,
                ``mean_stress`` is out of range, or ``criterion`` is unknown.
        """
        self._validate_cycle_stress(stress_amplitude, mean_stress, criterion)
        sigma_ar = self._equivalent_reversed_stress(
            stress_amplitude, mean_stress, criterion)
        if sigma_ar <= self.endurance_limit:
            return math.inf
        return (sigma_ar / self.sn_a) ** (1 / self.sn_b)

    def fatigue_safety_factor(self, stress_amplitude, mean_stress=0.0,
                              criterion="goodman"):
        """
        Infinite-life fatigue safety factor for a mean-stress criterion.

        Closed forms (Shigley Ch. 6):
        - goodman (6-46): nf = 1/(sigma_a/Se + sigma_m/Sut)
        - soderberg (6-45): nf = 1/(sigma_a/Se + sigma_m/Sy)
        - gerber (6-48): parabolic, uses Sut
        - asme_elliptic (6-50): nf = 1/sqrt((sigma_a/Se)^2 + (sigma_m/Sy)^2)

        Args:
            stress_amplitude (float): Alternating stress sigma_a in MPa.
                Must be strictly positive.
            mean_stress (float): Mean stress sigma_m in MPa (default: 0).
                Must be within [0, Sut) for goodman/gerber or [0, Sy) for
                soderberg/asme_elliptic.
            criterion (str): Mean-stress criterion (default: "goodman").
                One of ``_FATIGUE_CRITERIA``.

        Returns:
            float: Safety factor against infinite-life fatigue failure.

        Raises:
            ValueError: If ``stress_amplitude`` is not strictly positive,
                ``mean_stress`` is out of range, or ``criterion`` is unknown.
        """
        self._validate_cycle_stress(stress_amplitude, mean_stress, criterion)
        return self._fatigue_nf(stress_amplitude, mean_stress, criterion)

    def goodman_safety_factor(self, stress_amplitude, mean_stress=0.0):
        """
        Fatigue safety factor per the modified Goodman criterion (Eq. 6-46).

        Alias for ``fatigue_safety_factor(..., criterion="goodman")``, kept
        for backward compatibility.

        Args:
            stress_amplitude (float): Alternating stress sigma_a in MPa.
                Must be strictly positive.
            mean_stress (float): Mean stress sigma_m in MPa (default: 0).
                Must be within [0, Sut).

        Returns:
            float: Safety factor against infinite-life fatigue failure.

        Raises:
            ValueError: If ``stress_amplitude`` is not strictly positive or
                ``mean_stress`` is outside [0, Sut).
        """
        return self.fatigue_safety_factor(stress_amplitude, mean_stress,
                                          "goodman")

    def langer_safety_factor(self, stress_amplitude, mean_stress=0.0):
        """
        First-cycle static-yield safety factor (Langer line, Eq. 6-49).

        n = Sy / (sigma_a + sigma_m). This guards against yielding on the
        first load application, independent of the fatigue criterion.

        Args:
            stress_amplitude (float): Alternating stress sigma_a in MPa.
                Must be strictly positive.
            mean_stress (float): Mean stress sigma_m in MPa (default: 0).
                Must be non-negative.

        Returns:
            float: Safety factor against first-cycle yielding.

        Raises:
            ValueError: If ``stress_amplitude`` is not strictly positive or
                ``mean_stress`` is negative.
        """
        if stress_amplitude <= 0:
            raise ValueError("Stress amplitude must be strictly positive")
        if mean_stress < 0:
            raise ValueError("Mean stress must be non-negative")
        return self.yield_strength / (stress_amplitude + mean_stress)

    def governing_safety_factor(self, stress_amplitude, mean_stress=0.0,
                                criterion="goodman"):
        """
        Governing safety factor: the smaller of fatigue and Langer yield.

        Design practice caps the fatigue line with the first-cycle Langer
        yield line, so the governing factor is
        ``min(fatigue_safety_factor(...), langer_safety_factor(...))``.

        Args:
            stress_amplitude (float): Alternating stress sigma_a in MPa.
                Must be strictly positive.
            mean_stress (float): Mean stress sigma_m in MPa (default: 0).
            criterion (str): Fatigue criterion (default: "goodman"). One of
                ``_FATIGUE_CRITERIA``.

        Returns:
            tuple: ``(nf, mode)`` where ``nf`` is the governing safety
            factor and ``mode`` is ``"yield"`` if the Langer line governs,
            else ``"fatigue"``.

        Raises:
            ValueError: If any input fails validation or ``criterion`` is
                unknown.
        """
        fatigue = self.fatigue_safety_factor(stress_amplitude, mean_stress,
                                             criterion)
        yielding = self.langer_safety_factor(stress_amplitude, mean_stress)
        if yielding < fatigue:
            return yielding, "yield"
        return fatigue, "fatigue"

    # ---- Mean-stress helpers (pure; shared by cycles_to_failure) ----

    def _mean_strength(self, criterion):
        """float: Mean-axis intercept for a criterion (Sut or Sy), in MPa.

        Goodman and Gerber intercept the mean axis at Sut; Soderberg and
        ASME-elliptic intercept it at Sy. Raises ValueError on an unknown
        criterion.
        """
        if criterion not in _FATIGUE_CRITERIA:
            raise ValueError(
                f"Unknown criterion {criterion!r}. "
                f"Available: {', '.join(_FATIGUE_CRITERIA)}"
            )
        if criterion in ("soderberg", "asme_elliptic"):
            return self.yield_strength
        return self.ultimate_strength

    def _validate_cycle_stress(self, stress_amplitude, mean_stress,
                               criterion="goodman"):
        """Validate a (sigma_a, sigma_m) pair for a criterion.

        Checks sigma_a > 0 and 0 <= sigma_m < the criterion's mean-axis
        intercept (Sut for goodman/gerber, Sy for soderberg/asme_elliptic).
        Raises ValueError (also on an unknown criterion).
        """
        limit = self._mean_strength(criterion)
        name = "Sy" if limit == self.yield_strength else "Sut"
        if stress_amplitude <= 0:
            raise ValueError("Stress amplitude must be strictly positive")
        if not 0 <= mean_stress < limit:
            raise ValueError(
                f"Mean stress must be within [0, {name}) "
                f"({name} = {limit} MPa) for the {criterion} criterion"
            )

    # ---- Integration with MechaElement ----

    @property
    def properties_si(self):
        """dict: SI (Pa) property dict for :class:`MechaElement` consumers.

        Copy of the base database row with ``yield_strength`` and
        ``ultimate_strength`` overwritten from this instance's MPa values.
        """
        props = dict(get_material_properties(self.base_material))
        props["yield_strength"] = self.yield_strength * 1e6
        props["ultimate_strength"] = self.ultimate_strength * 1e6
        return props

    # ---- Plots ----

    @staticmethod
    def _pyplot(feature):
        """Lazily import matplotlib.pyplot, raising if absent."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                f"matplotlib is required for {feature}; "
                "install it with 'pip install matplotlib'"
            )
        return plt

    def plot_sn_curve(self, load_cases=(), show=True, ax=None):
        """
        Plot the S-N (Woehler) diagram, optionally with load cases.

        Finite-life line S = a*N^b from 10^3 cycles down to the endurance
        limit, then the horizontal Se asymptote. Each finite-life load
        case in ``load_cases`` is marked at (N_i, sigma_ar_i).

        Args:
            load_cases (list): Optional load-case dicts (``stress_amplitude``,
                ``mean_stress``, ``label``) to annotate — e.g. a component's
                ``load_cases`` (default: none).
            show (bool): Call ``plt.show()`` (default: True).
            ax (matplotlib.axes.Axes): Optional axes to draw into.

        Returns:
            matplotlib.figure.Figure: The figure drawn into.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        plt = self._pyplot("plot_sn_curve")
        if ax is None:
            _, ax = plt.subplots()

        se = self.endurance_limit
        n_knee = (se / self.sn_a) ** (1 / self.sn_b)
        n_line = [10 ** (3 + i * (math.log10(n_knee) - 3) / 100)
                  for i in range(101)]
        ax.loglog(n_line, [self.sn_a * n ** self.sn_b for n in n_line],
                  color="tab:blue", label="S-N curve")
        ax.loglog([n_knee, 10 * n_knee], [se, se], color="tab:blue",
                  linestyle="--", label=f"Se = {se:.0f} MPa")

        for i, case in enumerate(load_cases):
            n_failure = self.cycles_to_failure(case["stress_amplitude"],
                                               case["mean_stress"])
            if not math.isfinite(n_failure):
                continue
            sigma_ar = self.sn_a * n_failure ** self.sn_b
            label = case["label"] or f"case {i}"
            ax.plot(n_failure, sigma_ar, "o", color="tab:red")
            ax.annotate(label, (n_failure, sigma_ar),
                        textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel("Cycles to failure N")
        ax.set_ylabel("Stress amplitude (MPa)")
        ax.set_title(f"Woehler diagram — {self.name or self.base_material}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        if show:
            plt.show()
        return ax.figure

    def plot_goodman(self, load_cases=(), show=True, ax=None):
        """
        Plot the Goodman diagram, optionally with load cases.

        Modified Goodman line from (0, Se) to (Sut, 0), Langer yield line
        from (0, Sy) to (Sy, 0), and each load case in ``load_cases`` at
        (sigma_m, sigma_a).

        Args:
            load_cases (list): Optional load-case dicts (``stress_amplitude``,
                ``mean_stress``, ``label``) to annotate (default: none).
            show (bool): Call ``plt.show()`` (default: True).
            ax (matplotlib.axes.Axes): Optional axes to draw into.

        Returns:
            matplotlib.figure.Figure: The figure drawn into.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        plt = self._pyplot("plot_goodman")
        if ax is None:
            _, ax = plt.subplots()

        se, sut, sy = (self.endurance_limit, self.ultimate_strength,
                       self.yield_strength)
        ax.plot([0, sut], [se, 0], color="tab:blue", label="Goodman line")
        ax.plot([0, sy], [sy, 0], color="tab:orange", linestyle="--",
                label="Langer (yield) line")

        for i, case in enumerate(load_cases):
            label = case["label"] or f"case {i}"
            ax.plot(case["mean_stress"], case["stress_amplitude"], "o",
                    color="tab:red")
            ax.annotate(label,
                        (case["mean_stress"], case["stress_amplitude"]),
                        textcoords="offset points", xytext=(5, 5))

        ax.set_xlabel("Mean stress (MPa)")
        ax.set_ylabel("Stress amplitude (MPa)")
        ax.set_title(f"Goodman diagram — {self.name or self.base_material}")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if show:
            plt.show()
        return ax.figure

    def __repr__(self):
        return (
            f"Material(name={self.name!r}, "
            f"base_material={self.base_material!r}, "
            f"ultimate_strength={self.ultimate_strength}, "
            f"yield_strength={self.yield_strength})"
        )


class CyclicLoadCases:
    """Component-side cyclic load spectrum and Miner's-rule accumulation.

    Mixed into a mechanical element (Shaft, Bolt, ...) that owns a material,
    this holds the *component's* own cyclic load cases and evaluates
    cumulative fatigue damage against the material's S-N curve (Shigley
    Ch. 6). The spectrum belongs to the component, not the material: two
    shafts of the same steel have different duty cycles, so the shared
    :class:`Material` is never mutated by load-case bookkeeping.

    The host must expose ``self.material`` (a database name or a
    :class:`Material`); a bare name is wrapped in a default ``Material`` for
    its S-N curve. Units: stresses in MPa, cycles dimensionless.
    """

    @property
    def _cases(self):
        """list: backing load-case list, created lazily on first access."""
        if not hasattr(self, "_load_cases"):
            self._load_cases = []
        return self._load_cases

    def _fatigue_material(self):
        """Material: this component's material as a fatigue-capable Material.

        A :class:`Material` is used as-is; a database name is wrapped in a
        default (machined, kb = 1) ``Material``. Pass a configured
        ``Material`` for a meaningful endurance limit.
        """
        if isinstance(self.material, Material):
            return self.material
        return Material(self.material)

    @property
    def load_cases(self):
        """list: Copy of the registered load-case dicts."""
        return [dict(case) for case in self._cases]

    def _find_load_case(self, index):
        """Return the list position for an int index or str label."""
        cases = self._cases
        if isinstance(index, str):
            for i, case in enumerate(cases):
                if case["label"] == index:
                    return i
            raise ValueError(f"No load case with label {index!r}")
        try:
            cases[index]
        except IndexError:
            raise ValueError(
                f"Load case index {index} out of range ({len(cases)} cases)"
            )
        return index

    def add_load_case(self, stress_amplitude, mean_stress=0.0, cycles=1,
                      label=None):
        """
        Register a cyclic load case on this component for Miner analysis.

        Args:
            stress_amplitude (float): Alternating stress sigma_a in MPa.
                Must be strictly positive.
            mean_stress (float): Mean stress sigma_m in MPa (default: 0).
                Must be within [0, Sut).
            cycles (float): Applied cycles n for Miner's rule (default: 1).
                Must be strictly positive.
            label (str): Optional label to edit/remove the case by name.

        Returns:
            The component itself (``self``), so calls can be chained.

        Raises:
            ValueError: If any argument fails its validation above.
        """
        self._fatigue_material()._validate_cycle_stress(
            stress_amplitude, mean_stress)
        if cycles <= 0:
            raise ValueError("Cycles must be strictly positive")
        self._cases.append({
            "stress_amplitude": stress_amplitude,
            "mean_stress": mean_stress,
            "cycles": cycles,
            "label": label,
        })
        return self

    def edit_load_case(self, index, **kwargs):
        """
        Edit an existing load case, revalidating the result.

        Args:
            index (int or str): List position or label of the case.
            **kwargs: Any of ``stress_amplitude``, ``mean_stress``,
                ``cycles``, ``label``.

        Returns:
            The component itself (``self``), so calls can be chained.

        Raises:
            ValueError: If the case is not found, an unknown field is
                given, or the edited case fails validation.
        """
        i = self._find_load_case(index)
        unknown = set(kwargs) - set(_LOAD_CASE_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown load case field(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(_LOAD_CASE_KEYS)}"
            )
        merged = {**self._cases[i], **kwargs}
        self._fatigue_material()._validate_cycle_stress(
            merged["stress_amplitude"], merged["mean_stress"])
        if merged["cycles"] <= 0:
            raise ValueError("Cycles must be strictly positive")
        self._cases[i] = merged
        return self

    def remove_load_case(self, index):
        """
        Remove a load case by list position or label.

        Args:
            index (int or str): List position or label of the case.

        Returns:
            The component itself (``self``), so calls can be chained.

        Raises:
            ValueError: If the case is not found.
        """
        del self._cases[self._find_load_case(index)]
        return self

    def miner_damage(self, criterion="goodman"):
        """
        Cumulative fatigue damage per Miner's rule (Eq. 6-58).

        D = sum(n_i / N_i) over this component's load cases against its
        material's S-N curve; cases at or below the endurance limit
        contribute 0. Failure is predicted at D >= 1.

        Args:
            criterion (str): Mean-stress criterion used for each case's
                cycles to failure (default: "goodman"). One of
                ``_FATIGUE_CRITERIA``.

        Returns:
            float: Damage fraction for one pass through the load cases.

        Raises:
            ValueError: If ``criterion`` is unknown or a stored case is out
                of range for it (e.g. sigma_m >= Sy under soderberg).
        """
        material = self._fatigue_material()
        damage = 0.0
        for case in self._cases:
            n_failure = material.cycles_to_failure(
                case["stress_amplitude"], case["mean_stress"], criterion)
            if math.isfinite(n_failure):
                damage += case["cycles"] / n_failure
        return damage

    def remaining_life(self, criterion="goodman"):
        """
        Repetitions of the whole load-case block until failure.

        Args:
            criterion (str): Mean-stress criterion (default: "goodman").
                One of ``_FATIGUE_CRITERIA``.

        Returns:
            float: 1 / miner_damage(); ``math.inf`` if every case is below
            the endurance limit (zero damage).

        Raises:
            ValueError: If ``criterion`` is unknown or a stored case is out
                of range for it.
        """
        damage = self.miner_damage(criterion)
        if damage == 0:
            return math.inf
        return 1 / damage

    def plot_sn_curve(self, show=True, ax=None):
        """Plot the material S-N diagram annotated with this component's
        load cases (see :meth:`Material.plot_sn_curve`)."""
        return self._fatigue_material().plot_sn_curve(
            load_cases=self.load_cases, show=show, ax=ax)

    def plot_goodman(self, show=True, ax=None):
        """Plot the material Goodman diagram annotated with this component's
        load cases (see :meth:`Material.plot_goodman`)."""
        return self._fatigue_material().plot_goodman(
            load_cases=self.load_cases, show=show, ax=ax)
