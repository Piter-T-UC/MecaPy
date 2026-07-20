"""Material database, properties and fatigue analysis for MecaPy.

Units convention: the underlying database (``constants.MATERIALS``) is strict
SI (Pa, kg/m^3, J/(kg*K)). The :class:`Material` class API works entirely in
**MPa** for strengths, **mm** for the size (diameter) input and **degrees
Celsius** for temperature, converting from the Pa database at the boundary.
Fatigue relations follow Shigley's Mechanical Engineering Design, Ch. 6.
"""

import math

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
LOAD_FACTORS = {"bending": 1.0, "axial": 0.85, "torsion": 0.59}

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

# Sut below which the fatigue-strength fraction f is taken as 0.9
# (70 kpsi, Shigley Fig. 6-18).
_F_FRACTION_SUT_MIN = 482.6  # MPa
_MPA_PER_KPSI = constants.PSI_TO_MPA * 1e3

# Validity range of the rotating-size factor kb (Shigley Eq. 6-20), mm.
_KB_DIAMETER_MIN = 2.79
_KB_DIAMETER_MAX = 254.0

_LOAD_CASE_KEYS = ("stress_amplitude", "mean_stress", "cycles", "label")


class Material:
    """
    Material with static strength data and Shigley Ch. 6 fatigue analysis.

    Strengths default from the material database (converted Pa -> MPa) and
    can be overridden explicitly. The endurance limit follows the Marin
    equation Se = ka*kb*kc*kd*ke*Se' with Se' estimated from Sut (steel
    correlation). Cyclic load cases are evaluated with the Goodman
    criterion, accumulated with Miner's rule, and plotted on S-N (Woehler)
    and Goodman diagrams.

    Every Marin input (surface finish, loading, diameter, temperature,
    reliability) and both strengths are settable; all derived quantities
    (ka..ke, Se, S-N coefficients, damage) recompute automatically.

    Attributes:
        base_material (str): Database material this instance is based on
            (fixed at construction; supplies density, moduli, etc.).
        ultimate_strength (float): Ultimate tensile strength Sut in MPa.
        yield_strength (float): Yield strength Sy in MPa.
        surface_finish (str): Surface condition for ka (key of
            ``SURFACE_FINISH_FACTORS``).
        loading (str): Load type for kb/kc ("bending", "axial", "torsion").
        diameter (float or None): Rotating-section diameter in mm for kb
            (None -> kb = 1).
        temperature (float): Operating temperature in deg C for kd.
        reliability (float): Survival probability for ke (key of
            ``RELIABILITY_FACTORS``).
        name (str): Optional identifier.
    """

    def __init__(self, base_material="steel", ultimate_strength=None,
                 yield_strength=None, surface_finish="machined",
                 loading="bending", diameter=None, temperature=20.0,
                 reliability=0.5, name=None):
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
            diameter (float): Optional rotating-section diameter in mm for
                the size factor kb. Must be within [2.79, 254] mm
                (Shigley Eq. 6-20 validity range). None gives kb = 1.
            temperature (float): Operating temperature in deg C (default:
                20). Must be within [20, 600] (Table 6-4 range).
            reliability (float): Survival probability (default: 0.5, i.e.
                ke = 1). Must be a key of ``RELIABILITY_FACTORS``.
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
        self.temperature = temperature
        self.reliability = reliability
        self._load_cases = []

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
        if value not in RELIABILITY_FACTORS:
            available = ", ".join(str(r) for r in sorted(RELIABILITY_FACTORS))
            raise ValueError(
                f"Unknown reliability {value!r}. Available: {available}"
            )
        self._reliability = value

    # ---- Marin factors and endurance limit (recomputed on access) ----

    @property
    def ka(self):
        """float: Marin surface-condition factor, a*Sut^b (Table 6-2)."""
        a, b = SURFACE_FINISH_FACTORS[self.surface_finish]
        return a * self.ultimate_strength ** b

    @property
    def kb(self):
        """float: Marin size factor (Eq. 6-19/6-20).

        1.0 for axial loading or when no diameter is set; otherwise
        (d/7.62)^-0.107 for d <= 51 mm and 1.51*d^-0.157 above.
        """
        if self.loading == "axial" or self.diameter is None:
            return 1.0
        if self.diameter <= 51:
            return (self.diameter / 7.62) ** -0.107
        return 1.51 * self.diameter ** -0.157

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
    def ke(self):
        """float: Marin reliability factor (Table 6-5)."""
        return RELIABILITY_FACTORS[self.reliability]

    @property
    def se_prime(self):
        """float: Unmodified endurance limit Se' in MPa.

        Steel correlation (Eq. 6-8): 0.5*Sut for Sut <= 1400 MPa, else
        700 MPa. For non-ferrous materials this is an approximation only —
        many (e.g. aluminum) have no true endurance limit.
        """
        if self.ultimate_strength <= 1400:
            return 0.5 * self.ultimate_strength
        return 700.0

    @property
    def endurance_limit(self):
        """float: Corrected endurance limit Se in MPa.

        Marin equation (Eq. 6-17): Se = ka*kb*kc*kd*ke*Se'. The
        miscellaneous-effects factor kf is taken as 1.
        """
        return self.ka * self.kb * self.kc * self.kd * self.ke * self.se_prime

    # ---- S-N (Woehler) curve ----

    @property
    def f(self):
        """float: Fatigue-strength fraction of Sut at 10^3 cycles.

        Fig. 6-18 fit: 0.9 below 482.6 MPa (70 kpsi), else the quadratic
        1.06 - 2.8e-3*Sut + 6.9e-6*Sut^2 with Sut in kpsi.
        """
        if self.ultimate_strength < _F_FRACTION_SUT_MIN:
            return 0.9
        sut_kpsi = self.ultimate_strength / _MPA_PER_KPSI
        return 1.06 - 2.8e-3 * sut_kpsi + 6.9e-6 * sut_kpsi ** 2

    @property
    def sn_a(self):
        """float: S-N curve coefficient a in MPa, (f*Sut)^2/Se (Eq. 6-14)."""
        return (self.f * self.ultimate_strength) ** 2 / self.endurance_limit

    @property
    def sn_b(self):
        """float: S-N curve exponent b, -(1/3)*log10(f*Sut/Se) (Eq. 6-15)."""
        return -math.log10(self.f * self.ultimate_strength
                           / self.endurance_limit) / 3

    def cycles_to_failure(self, stress_amplitude, mean_stress=0.0):
        """
        Cycles to failure N for a cyclic stress (Goodman + S-N, Eq. 6-16).

        The mean stress is folded into an equivalent fully reversed
        amplitude with the Goodman line, sigma_ar = sigma_a/(1 - sigma_m/Sut),
        then N = (sigma_ar/a)^(1/b).

        Args:
            stress_amplitude (float): Alternating stress sigma_a in MPa.
                Must be strictly positive.
            mean_stress (float): Mean stress sigma_m in MPa (default: 0).
                Must be within [0, Sut) — the Goodman line applies to
                tensile mean stress.

        Returns:
            float: Cycles to failure; ``math.inf`` if the equivalent
            amplitude is at or below the endurance limit.

        Raises:
            ValueError: If ``stress_amplitude`` is not strictly positive or
                ``mean_stress`` is outside [0, Sut).
        """
        self._validate_cycle_stress(stress_amplitude, mean_stress)
        sigma_ar = stress_amplitude / (1 - mean_stress / self.ultimate_strength)
        if sigma_ar <= self.endurance_limit:
            return math.inf
        return (sigma_ar / self.sn_a) ** (1 / self.sn_b)

    def goodman_safety_factor(self, stress_amplitude, mean_stress=0.0):
        """
        Fatigue safety factor per the modified Goodman criterion (Eq. 6-46).

        nf = 1 / (sigma_a/Se + sigma_m/Sut).

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
        self._validate_cycle_stress(stress_amplitude, mean_stress)
        return 1 / (stress_amplitude / self.endurance_limit
                    + mean_stress / self.ultimate_strength)

    # ---- Cyclic load cases, Miner's rule ----

    def _validate_cycle_stress(self, stress_amplitude, mean_stress):
        """Validate a (sigma_a, sigma_m) pair, raising ValueError."""
        if stress_amplitude <= 0:
            raise ValueError("Stress amplitude must be strictly positive")
        if not 0 <= mean_stress < self.ultimate_strength:
            raise ValueError(
                "Mean stress must be within [0, Sut) "
                f"(Sut = {self.ultimate_strength} MPa)"
            )

    def _find_load_case(self, index):
        """Return the list position for an int index or str label."""
        if isinstance(index, str):
            for i, case in enumerate(self._load_cases):
                if case["label"] == index:
                    return i
            raise ValueError(f"No load case with label {index!r}")
        try:
            self._load_cases[index]
        except IndexError:
            raise ValueError(
                f"Load case index {index} out of range "
                f"({len(self._load_cases)} cases)"
            )
        return index

    @property
    def load_cases(self):
        """list: Copy of the registered load-case dicts."""
        return [dict(case) for case in self._load_cases]

    def add_load_case(self, stress_amplitude, mean_stress=0.0, cycles=1,
                      label=None):
        """
        Register a cyclic load case for Goodman/Miner analysis.

        Args:
            stress_amplitude (float): Alternating stress sigma_a in MPa.
                Must be strictly positive.
            mean_stress (float): Mean stress sigma_m in MPa (default: 0).
                Must be within [0, Sut).
            cycles (float): Applied cycles n for Miner's rule (default: 1).
                Must be strictly positive.
            label (str): Optional label to edit/remove the case by name.

        Returns:
            Material: ``self``, so calls can be chained.

        Raises:
            ValueError: If any argument fails its validation above.
        """
        self._validate_cycle_stress(stress_amplitude, mean_stress)
        if cycles <= 0:
            raise ValueError("Cycles must be strictly positive")
        self._load_cases.append({
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
            Material: ``self``, so calls can be chained.

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
        merged = {**self._load_cases[i], **kwargs}
        self._validate_cycle_stress(merged["stress_amplitude"],
                                    merged["mean_stress"])
        if merged["cycles"] <= 0:
            raise ValueError("Cycles must be strictly positive")
        self._load_cases[i] = merged
        return self

    def remove_load_case(self, index):
        """
        Remove a load case by list position or label.

        Args:
            index (int or str): List position or label of the case.

        Returns:
            Material: ``self``, so calls can be chained.

        Raises:
            ValueError: If the case is not found.
        """
        del self._load_cases[self._find_load_case(index)]
        return self

    def miner_damage(self):
        """
        Cumulative fatigue damage per Miner's rule (Eq. 6-58).

        D = sum(n_i / N_i) over the registered load cases; cases at or
        below the endurance limit contribute 0. Failure is predicted at
        D >= 1.

        Returns:
            float: Damage fraction for one pass through the load cases.
        """
        damage = 0.0
        for case in self._load_cases:
            n_failure = self.cycles_to_failure(case["stress_amplitude"],
                                              case["mean_stress"])
            if math.isfinite(n_failure):
                damage += case["cycles"] / n_failure
        return damage

    def remaining_life(self):
        """
        Repetitions of the whole load-case block until failure.

        Returns:
            float: 1 / miner_damage(); ``math.inf`` if every case is
            below the endurance limit (zero damage).
        """
        damage = self.miner_damage()
        if damage == 0:
            return math.inf
        return 1 / damage

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

    def plot_sn_curve(self, show=True, ax=None):
        """
        Plot the S-N (Woehler) diagram with the registered load cases.

        Finite-life line S = a*N^b from 10^3 cycles down to the endurance
        limit, then the horizontal Se asymptote. Each finite-life load
        case is marked at (N_i, sigma_ar_i).

        Args:
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

        for i, case in enumerate(self._load_cases):
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

    def plot_goodman(self, show=True, ax=None):
        """
        Plot the Goodman diagram with the registered load cases.

        Modified Goodman line from (0, Se) to (Sut, 0), Langer yield line
        from (0, Sy) to (Sy, 0), and each load case at (sigma_m, sigma_a).

        Args:
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

        for i, case in enumerate(self._load_cases):
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
            f"Material(base_material={self.base_material!r}, "
            f"ultimate_strength={self.ultimate_strength}, "
            f"yield_strength={self.yield_strength})"
        )
