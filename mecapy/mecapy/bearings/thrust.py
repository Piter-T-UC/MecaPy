"""Hydrodynamic fixed-incline thrust bearing (tapered-land thrust pad).

A thrust collar running against a ring of tapered-land pads.  Each pad is
treated as a one-dimensional plane slider, whose Reynolds solution is
exact in closed form — so unlike the journal-bearing charts nothing here
is digitized:

::

    a  = h1 / h2                                    taper ratio
    W' = 6 mu U B^2 / h2^2 * [ln a - 2(a-1)/(a+1)] / (a-1)^2
    F' = mu U B / h2 * [4 ln a/(a-1) - 6/(a+1)]
    q  = U h2 a / (a + 1)

with W' and F' per unit of pad width.  The load coefficient peaks at
a ~ 2.19 (:data:`OPTIMUM_TAPER_RATIO`), which is why tapered lands are
cut to roughly that ratio.

The pads are rectangular in the sliding direction and are evaluated at
the mean radius, the standard sector-pad approximation: side leakage is
neglected, so the load capacity is optimistic for short, wide pads.

Units: geometry in mm, speed in rev/s, load in N, viscosity in mPa*s,
pressure in MPa, torque in N*mm, power in W.

Dimensional inputs additionally accept a ``pint.Quantity``, which is
converted to the documented unit at the boundary; plain floats are
assumed to be in that unit already and behave exactly as before.
"""

import math

from ..base import MechaElement
from ..utils.units import to_magnitude
from .lubrication_data import viscosity as sae_viscosity

#: Taper ratio h1/h2 maximizing the load coefficient of a plane slider.
#: Found by maximizing [ln a - 2(a-1)/(a+1)]/(a-1)^2; the classic value
#: quoted in the literature is 2.2.
OPTIMUM_TAPER_RATIO = 2.1887


def load_coefficient(taper_ratio):
    """Dimensionless load capacity of a plane slider.

    ::

        Kw = 6 * [ln a - 2 (a - 1)/(a + 1)] / (a - 1)^2

    so that ``W' = Kw * mu * U * B^2 / h2^2`` per unit width.

    Args:
        taper_ratio (float): Film ratio a = h1/h2, strictly above 1.

    Returns:
        float: Load coefficient Kw (dimensionless).

    Raises:
        ValueError: If the taper ratio is not greater than 1.
    """
    if taper_ratio <= 1.0:
        raise ValueError("Taper ratio must be strictly greater than 1")
    return (
        6.0
        * (math.log(taper_ratio) - 2.0 * (taper_ratio - 1.0) / (taper_ratio + 1.0))
        / (taper_ratio - 1.0) ** 2
    )


def friction_coefficient_factor(taper_ratio):
    """Dimensionless friction force of a plane slider.

    ::

        Kf = 4 ln a / (a - 1) - 6 / (a + 1)

    so that ``F' = Kf * mu * U * B / h2`` per unit width.

    Args:
        taper_ratio (float): Film ratio a = h1/h2, strictly above 1.

    Returns:
        float: Friction coefficient factor Kf (dimensionless).

    Raises:
        ValueError: If the taper ratio is not greater than 1.
    """
    if taper_ratio <= 1.0:
        raise ValueError("Taper ratio must be strictly greater than 1")
    return 4.0 * math.log(taper_ratio) / (taper_ratio - 1.0) - 6.0 / (taper_ratio + 1.0)


class ThrustBearing(MechaElement):
    """
    Hydrodynamic fixed-incline (tapered-land) thrust bearing.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`.

    Attributes:
        inner_radius (float): Inner radius of the pad ring in mm.
        outer_radius (float): Outer radius of the pad ring in mm.
        n_pads (int): Number of pads around the collar.
        speed (float): Collar speed in rev/s.
        load (float): Total axial (thrust) load in N.
        taper_ratio (float): Film ratio a = h1/h2.
        pad_fraction (float): Fraction of the circumference the pads
            occupy (the remainder is the oil-feed groove).
        viscosity (float): Lubricant absolute viscosity in mPa*s.
    """

    def __init__(
        self,
        inner_radius,
        outer_radius,
        n_pads,
        speed,
        load,
        taper_ratio=OPTIMUM_TAPER_RATIO,
        pad_fraction=0.8,
        viscosity=None,
        sae_grade=None,
        temperature=None,
        material="steel",
        name=None,
    ):
        """
        Initialize a ThrustBearing object.

        The lubricant is specified either directly with ``viscosity`` or
        by ``sae_grade`` and ``temperature`` — exactly one of the two,
        the same convention as
        :class:`~mecapy.bearings.journal.JournalBearing`.

        Args:
            inner_radius (float): Inner radius of the pad ring in mm.
            outer_radius (float): Outer radius in mm (must exceed the
                inner radius).
            n_pads (int): Number of pads, an integer >= 1.
            speed (float): Collar speed in rev/s (or a pint.Quantity of
                rotational speed).
            load (float): Total axial load in N (or a pint.Quantity).
            taper_ratio (float): Film ratio a = h1/h2 (default: the
                load-optimal 2.19).
            pad_fraction (float): Fraction of the circumference occupied
                by pad, in (0, 1] (default: 0.8).
            viscosity (float): Absolute viscosity in mPa*s.
            sae_grade (int): SAE oil grade (10-60), with ``temperature``.
            temperature (float): Oil temperature in degrees Celsius, with
                ``sae_grade``.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the bearing.

        Raises:
            ValueError: For non-positive or inverted geometry, a
                non-integer or non-positive pad count, a taper ratio not
                above 1, a pad fraction outside (0, 1], or an over- or
                under-specified lubricant.
        """
        super().__init__(name=name, material=material)
        if viscosity is not None and (sae_grade is not None or temperature is not None):
            raise ValueError(
                "Specify either viscosity or sae_grade+temperature, not both"
            )
        if viscosity is None:
            if sae_grade is None or temperature is None:
                raise ValueError(
                    "Lubricant unspecified: pass viscosity= or both "
                    "sae_grade= and temperature="
                )
            temperature = to_magnitude(temperature, "degC")
            viscosity = sae_viscosity(sae_grade, temperature)
        self._sae_grade = sae_grade
        self._film_temperature = temperature
        if not isinstance(n_pads, int) or n_pads < 1:
            raise ValueError("n_pads must be an integer >= 1")
        self.n_pads = n_pads
        if not 0 < pad_fraction <= 1:
            raise ValueError("Pad fraction must be in (0, 1]")
        self.pad_fraction = pad_fraction
        # Validating setters below, so mutating an input re-checks it.
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.speed = speed
        self.load = load
        self.taper_ratio = taper_ratio
        self.viscosity = viscosity

    # ---- Settable primary inputs (validate; never cache a derived value) ----

    @property
    def inner_radius(self):
        """float: Inner radius of the pad ring in mm."""
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, value):
        value = to_magnitude(value, "mm")
        if value <= 0:
            raise ValueError("Inner radius must be strictly positive")
        if value >= getattr(self, "_outer_radius", math.inf):
            raise ValueError("Inner radius must be smaller than the outer radius")
        self._inner_radius = value

    @property
    def outer_radius(self):
        """float: Outer radius of the pad ring in mm."""
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, value):
        value = to_magnitude(value, "mm")
        if value <= self._inner_radius:
            raise ValueError("Outer radius must be larger than the inner radius")
        self._outer_radius = value

    @property
    def speed(self):
        """float: Collar speed in rev/s."""
        return self._speed

    @speed.setter
    def speed(self, value):
        value = to_magnitude(value, "revolution / second")
        if value <= 0:
            raise ValueError("Speed must be strictly positive")
        self._speed = value

    @property
    def load(self):
        """float: Total axial load in N."""
        return self._load

    @load.setter
    def load(self, value):
        value = to_magnitude(value, "N")
        if value <= 0:
            raise ValueError("Load must be strictly positive")
        self._load = value

    @property
    def taper_ratio(self):
        """float: Film ratio a = h1/h2 (inlet over outlet film)."""
        return self._taper_ratio

    @taper_ratio.setter
    def taper_ratio(self, value):
        if value <= 1.0:
            raise ValueError("Taper ratio must be strictly greater than 1")
        self._taper_ratio = value

    @property
    def viscosity(self):
        """float: Absolute lubricant viscosity in mPa*s."""
        return self._viscosity

    @viscosity.setter
    def viscosity(self, value):
        value = to_magnitude(value, "mPa*s")
        if value <= 0:
            raise ValueError("Viscosity must be strictly positive")
        self._viscosity = value

    # ---- Pad geometry and kinematics ----

    @property
    def mean_radius(self):
        """float: Mean radius of the pad ring in mm."""
        return 0.5 * (self.inner_radius + self.outer_radius)

    @property
    def pad_length(self):
        """float: Radial extent of one pad in mm (the slider's width)."""
        return self.outer_radius - self.inner_radius

    @property
    def pad_width(self):
        """float: Circumferential extent of one pad at the mean radius, mm.

        ``B = pad_fraction * 2 pi r_mean / n_pads`` — the sliding
        direction, so this is the slider length B in the Reynolds
        solution.
        """
        return self.pad_fraction * 2.0 * math.pi * self.mean_radius / self.n_pads

    @property
    def pad_area(self):
        """float: Area of one pad in mm^2."""
        return self.pad_length * self.pad_width

    @property
    def sliding_velocity(self):
        """float: Surface speed at the mean radius in m/s."""
        return 2.0 * math.pi * self.mean_radius * self.speed / 1000.0

    @property
    def pad_load(self):
        """float: Axial load carried by one pad in N."""
        return self.load / self.n_pads

    @property
    def pressure(self):
        """float: Mean pad pressure in MPa (pad load over pad area)."""
        return self.pad_load / self.pad_area

    # ---- Film solution ----

    def film_thickness(self):
        """Minimum (outlet) film thickness h2 in mm.

        Inverts the plane-slider load relation, which is exactly
        quadratic in 1/h2, so no iteration is needed::

            h2 = sqrt(Kw * mu * U * B^2 * L / W_pad)

        Returns:
            float: Minimum film thickness h2 in mm.
        """
        # SI throughout, converted back to mm on the way out.
        viscosity_pa_s = self.viscosity * 1e-3
        width = self.pad_width / 1000.0
        length = self.pad_length / 1000.0
        coefficient = load_coefficient(self.taper_ratio)
        h2_squared = (
            coefficient
            * viscosity_pa_s
            * self.sliding_velocity
            * width**2
            * length
            / self.pad_load
        )
        return math.sqrt(h2_squared) * 1000.0

    def performance(self):
        """Full operating point of the bearing.

        Returns:
            dict: ``film_thickness`` h2 (mm), ``inlet_film`` h1 (mm),
            ``friction_force`` per pad (N), ``friction_torque`` (N*mm,
            whole bearing), ``friction_coefficient`` (friction over
            thrust), ``power_loss`` (W, whole bearing), ``flow`` per pad
            (mm^3/s), ``pressure`` mean pad pressure (MPa) and
            ``pmax`` peak film pressure (MPa).
        """
        h2 = self.film_thickness()
        h1 = self.taper_ratio * h2
        viscosity_pa_s = self.viscosity * 1e-3
        width = self.pad_width / 1000.0
        length = self.pad_length / 1000.0
        velocity = self.sliding_velocity
        friction = (
            friction_coefficient_factor(self.taper_ratio)
            * viscosity_pa_s
            * velocity
            * width
            * length
            / (h2 / 1000.0)
        )  # N per pad
        torque = friction * self.n_pads * self.mean_radius  # N*mm
        flow = (
            velocity
            * (h2 / 1000.0)
            * self.taper_ratio
            / (self.taper_ratio + 1.0)
            * length
        )  # m^3/s per pad
        profile = self.pressure_profile(n=201)
        return {
            "film_thickness": h2,
            "inlet_film": h1,
            "friction_force": friction,
            "friction_torque": torque,
            "friction_coefficient": friction * self.n_pads / self.load,
            "power_loss": friction * self.n_pads * velocity,
            "flow": flow * 1e9,  # m^3/s -> mm^3/s
            "pressure": self.pressure,
            "pmax": max(profile["pressure"]),
        }

    def pressure_profile(self, n=101):
        """Film pressure along the sliding direction of one pad.

        Solution of the 1-D Reynolds equation for a plane slider, with
        p = 0 at both ends::

            p(x) = 6 mu U / m * [1/h - h* / (2 h^2)] + C
            h(x) = h1 - m x,   m = (h1 - h2)/B,   h* = 2 h1 h2/(h1 + h2)

        Matplotlib-free, so it can be inspected and tested without a
        plotting backend.

        Args:
            n (int): Number of samples along the pad (default: 101).

        Returns:
            dict: ``x`` (mm from the inlet), ``film_thickness`` (mm) and
            ``pressure`` (MPa), each a list of length n.

        Raises:
            ValueError: If n is less than 2.
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        h2 = self.film_thickness()
        h1 = self.taper_ratio * h2
        width = self.pad_width
        slope = (h1 - h2) / width
        h_star = 2.0 * h1 * h2 / (h1 + h2)
        # mu in Pa*s, lengths in mm: mu*U/(m) with U in m/s and h in mm
        # gives Pa*s*m/s/mm = Pa*m/mm -> divide by 1e3 for MPa*mm/mm.
        scale = 6.0 * (self.viscosity * 1e-3) * self.sliding_velocity / slope
        offset = -(1.0 / h1 - h_star / (2.0 * h1**2))
        result = {"x": [], "film_thickness": [], "pressure": []}
        for index in range(n):
            x = width * index / (n - 1)
            h = h1 - slope * x
            pressure_pa = scale * (1.0 / h - h_star / (2.0 * h**2) + offset) * 1000.0
            result["x"].append(x)
            result["film_thickness"].append(h)
            result["pressure"].append(pressure_pa / 1e6)
        return result

    def optimum_taper_ratio(self):
        """Taper ratio that maximizes load capacity at a given film.

        Golden-section search on :func:`load_coefficient`; lands on
        :data:`OPTIMUM_TAPER_RATIO`.

        Returns:
            float: Optimal film ratio a = h1/h2.
        """
        low, high = 1.001, 10.0
        golden = 0.5 * (math.sqrt(5.0) - 1.0)
        for _ in range(200):
            probe_low = high - golden * (high - low)
            probe_high = low + golden * (high - low)
            if load_coefficient(probe_low) < load_coefficient(probe_high):
                low = probe_low
            else:
                high = probe_high
            if high - low < 1e-9:
                break
        return 0.5 * (low + high)

    def temperature_rise(self, density=870.0, specific_heat=1760.0):
        """Lubricant temperature rise, all friction heat carried by the oil.

        ::

            dT = H / (rho * cp * Q)

        with H the frictional power and Q the total flow through the
        pads.  The same closed-thermal-system assumption the journal
        bearing's rise makes.

        Note that the result depends only on the pad pressure: a thicker
        oil raises the friction and the flow by the same sqrt(mu), and
        speed cancels likewise, so dT is proportional to W/(B*L) alone —
        the thrust analogue of the journal bearing's dT = 8.30 P (...).

        Args:
            density (float): Lubricant density in kg/m^3 (default: 870).
            specific_heat (float): Specific heat in J/(kg*K)
                (default: 1760, typical mineral oil).

        Returns:
            float: Temperature rise in degrees Celsius.

        Raises:
            ValueError: For a non-positive density or specific heat.
        """
        if density <= 0:
            raise ValueError("Density must be strictly positive")
        if specific_heat <= 0:
            raise ValueError("Specific heat must be strictly positive")
        performance = self.performance()
        flow_m3_s = performance["flow"] * self.n_pads / 1e9
        return performance["power_loss"] / (density * specific_heat * flow_m3_s)

    # ---- Report and plot ----

    def describe(self):
        """
        Human-readable summary of geometry and operating point.

        The string is returned, not printed; use
        ``print(bearing.describe())``.

        Returns:
            str: Multi-line description, one quantity per line as
            ``label (symbol) = value unit``.
        """
        performance = self.performance()
        header = f"{self.__class__.__name__} geometry"
        if self.name:
            header += f" '{self.name}'"
        lines = [
            header,
            "=" * 40,
            f"inner radius (ri) = {self.inner_radius:.3f} mm",
            f"outer radius (ro) = {self.outer_radius:.3f} mm",
            f"mean radius (rm) = {self.mean_radius:.3f} mm",
            f"pad count (n) = {self.n_pads}",
            f"pad width (B) = {self.pad_width:.3f} mm",
            f"pad length (L) = {self.pad_length:.3f} mm",
            f"taper ratio (a) = {self.taper_ratio:.3f}",
            f"speed (N) = {self.speed:.3f} rev/s ({60.0 * self.speed:.1f} rpm)",
            f"sliding velocity (U) = {self.sliding_velocity:.3f} m/s",
            f"total load (W) = {self.load:.1f} N",
            f"mean pad pressure (p) = {self.pressure:.4f} MPa",
            f"viscosity (mu) = {self.viscosity:.2f} mPa*s",
            f"minimum film (h2) = {performance['film_thickness'] * 1000.0:.2f} um",
            f"inlet film (h1) = {performance['inlet_film'] * 1000.0:.2f} um",
            f"peak film pressure (pmax) = {performance['pmax']:.3f} MPa",
            f"friction coefficient (f) = {performance['friction_coefficient']:.5f}",
            f"friction torque (T) = {performance['friction_torque']:.1f} N*mm",
            f"power loss (H) = {performance['power_loss']:.1f} W",
            f"flow per pad (q) = {performance['flow']:.1f} mm^3/s",
            f"material = {self.material}",
        ]
        return "\n".join(lines)

    def plot_pressure(self, n=201, show=True, ax=None):
        """Plot the film pressure and thickness along one pad.

        Args:
            n (int): Number of samples along the pad (default: 201).
            show (bool): Call ``plt.show()`` before returning.
            ax (matplotlib.axes.Axes): Axes to draw on; a new figure is
                created when omitted.

        Returns:
            matplotlib.figure.Figure: The figure drawn on.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_pressure; "
                "install it with 'pip install matplotlib'"
            )
        profile = self.pressure_profile(n=n)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure
        ax.plot(
            profile["x"], profile["pressure"], color="#1d4ed8", label="film pressure"
        )
        ax.fill_between(profile["x"], profile["pressure"], color="#dbeafe")
        ax.set_xlabel("distance from pad inlet [mm]")
        ax.set_ylabel("film pressure [MPa]")
        film_axis = ax.twinx()
        film_axis.plot(
            profile["x"],
            [value * 1000.0 for value in profile["film_thickness"]],
            color="#6b7280",
            linestyle="--",
            label="film thickness",
        )
        film_axis.set_ylabel("film thickness [um]")
        ax.set_title(f"Tapered-land pad, a = {self.taper_ratio:.2f}")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left")
        if show:
            plt.show()
        return fig

    def __repr__(self):
        return (
            f"ThrustBearing(ri={self.inner_radius}, ro={self.outer_radius}, "
            f"n_pads={self.n_pads}, N={self.speed} rev/s)"
        )
