"""Boundary-lubricated plain (bushing) bearing analysis (Shigley sec. 12-15).

This is the regime a :class:`~mecapy.bearings.journal.JournalBearing`
cannot describe: at low speed, high load or with no pressurised oil
supply there is no hydrodynamic film to compute, and the bearing is
rated instead against the material's pressure, velocity and PV limits.
``JournalBearing.is_thick_film()`` returning False is the signal that a
bearing belongs here rather than there.

Units: geometry in mm, load in N, speed in rev/s, pressure in MPa,
rubbing velocity in m/s, PV in MPa*m/s, torque in N*mm, power in W.

Dimensional inputs additionally accept a ``pint.Quantity``, which is
converted to the documented unit at the boundary; plain floats are
assumed to be in that unit already and behave exactly as before.
"""

import math

from ..base import MechaElement
from ..utils.units import to_magnitude
from .bushing_data import DEFAULT_BUSHING_MU, get_bushing_material


class PlainBearing(MechaElement):
    """
    Boundary-lubricated plain bearing (sleeve bushing).

    Rates a bushing against the three limits that govern this regime
    (Shigley sec. 12-15): the projected-area pressure P, the rubbing
    velocity V, and their product PV, which stands in for the frictional
    heat the bushing has to shed per unit area.

    Inherits shared material and stress behaviour from
    :class:`~mecapy.base.MechaElement`.  Note that ``material`` is the
    structural material of the housing/shaft for the inherited stress
    helpers, while ``bushing_material`` selects the liner whose PV limits
    are checked — they are different things.

    Attributes:
        bore_diameter (float): Bushing bore (journal) diameter d in mm.
        length (float): Bushing length l in mm.
        load (float): Radial load W in N.
        speed (float): Journal speed N in rev/s.
        bushing_material (str or None): Liner material name.
        mu (float): Running friction coefficient.
    """

    def __init__(
        self,
        bore_diameter,
        length,
        load,
        speed,
        bushing_material=None,
        mu=None,
        material="steel",
        name=None,
    ):
        """
        Initialize a PlainBearing object.

        An explicit ``mu`` always wins over the value carried by
        ``bushing_material``; with neither, a mid-range boundary-lubricated
        default is used and the PV checks are unavailable (they need the
        material's limits).

        Args:
            bore_diameter (float): Bore diameter d in mm (or a
                pint.Quantity of length).
            length (float): Bushing length l in mm (or a pint.Quantity).
            load (float): Radial load W in N (or a pint.Quantity of force).
            speed (float): Journal speed N in rev/s (or a pint.Quantity
                of rotational speed).
            bushing_material (str): Liner material, a key of
                :data:`~mecapy.bearings.bushing_data.BUSHING_MATERIALS`.
            mu (float): Running friction coefficient, 0 < mu < 1.5.
            material (str): Structural material (default: "steel").
            name (str): Optional identifier for the bearing.

        Raises:
            ValueError: For non-positive geometry, load or speed, an
                unknown bushing material, or a friction coefficient
                outside (0, 1.5).
        """
        super().__init__(name=name, material=material)
        self._bushing_data = None
        if bushing_material is not None:
            self._bushing_data = get_bushing_material(bushing_material)
        self.bushing_material = bushing_material
        if mu is None:
            mu = self._bushing_data["mu"] if self._bushing_data else DEFAULT_BUSHING_MU
        if not 0 < mu < 1.5:
            raise ValueError("Friction coefficient must be in (0, 1.5)")
        self.mu = mu
        # Validating setters below, so mutating an input re-checks it.
        self.bore_diameter = bore_diameter
        self.length = length
        self.load = load
        self.speed = speed

    # ---- Settable primary inputs (validate; never cache a derived value) ----

    @property
    def bore_diameter(self):
        """float: Bore (journal) diameter d in mm."""
        return self._bore_diameter

    @bore_diameter.setter
    def bore_diameter(self, value):
        value = to_magnitude(value, "mm")
        if value <= 0:
            raise ValueError("Bore diameter must be strictly positive")
        self._bore_diameter = value

    @property
    def length(self):
        """float: Bushing length l in mm."""
        return self._length

    @length.setter
    def length(self, value):
        value = to_magnitude(value, "mm")
        if value <= 0:
            raise ValueError("Length must be strictly positive")
        self._length = value

    @property
    def load(self):
        """float: Radial load W in N."""
        return self._load

    @load.setter
    def load(self, value):
        value = to_magnitude(value, "N")
        if value <= 0:
            raise ValueError("Load must be strictly positive")
        self._load = value

    @property
    def speed(self):
        """float: Journal speed N in rev/s."""
        return self._speed

    @speed.setter
    def speed(self, value):
        value = to_magnitude(value, "revolution / second")
        if value <= 0:
            raise ValueError("Speed must be strictly positive")
        self._speed = value

    # ---- Operating point (pressure, velocity, PV) ----

    @property
    def speed_rpm(self):
        """float: Journal speed in rev/min."""
        return 60.0 * self.speed

    @property
    def l_over_d(self):
        """float: Length-to-diameter ratio l/d."""
        return self.length / self.bore_diameter

    @property
    def pressure(self):
        """Projected-area pressure P = W / (d * l) in MPa.

        ::

            P = W / (d * l)                        (Shigley sec. 12-15)
        """
        return self.load / (self.bore_diameter * self.length)

    @property
    def rubbing_velocity(self):
        """Surface speed at the bore V = pi * d * N in m/s.

        The bore diameter is in mm and the speed in rev/s, so the
        product is divided by 1000 once, here, to reach m/s.
        """
        return math.pi * self.bore_diameter * self.speed / 1000.0

    @property
    def pv(self):
        """PV product in MPa*m/s.

        The severity index of a boundary-lubricated bearing: pressure
        times rubbing velocity is proportional to the frictional power
        dissipated per unit of projected area, so it governs both wear
        rate and running temperature.
        """
        return self.pressure * self.rubbing_velocity

    def friction_force(self):
        """Friction force F = mu * W in N.

        Returns:
            float: Friction force in N.
        """
        return self.mu * self.load

    def friction_torque(self):
        """Friction torque T = mu * W * d / 2 in N*mm.

        Returns:
            float: Friction torque in N*mm.
        """
        return self.friction_force() * self.bore_diameter / 2.0

    def power_loss(self):
        """Frictional power H = mu * W * V in W.

        Returns:
            float: Power dissipated in the bushing in W.
        """
        return self.mu * self.load * self.rubbing_velocity

    def heat_flux(self):
        """Frictional power per unit of projected area in W/mm^2.

        Equivalent to ``mu * PV`` once the units are bridged; the form
        the housing has to conduct away.

        Returns:
            float: Heat flux in W/mm^2.
        """
        return self.power_loss() / (self.bore_diameter * self.length)

    # ---- Material limit checks ----

    def _require_material(self):
        if self._bushing_data is None:
            raise ValueError(
                "This check needs the liner limits; build the bearing with "
                "bushing_material= (see BUSHING_MATERIALS)"
            )

    def pressure_safety_factor(self):
        """Allowable over actual bearing pressure.

        Values greater than 1 mean the bushing is within its pressure
        rating.  Named ``pressure_safety_factor`` rather than
        ``safety_factor`` so it never shadows the inherited Pa-based
        :meth:`~mecapy.base.MechaElement.safety_factor`.

        Returns:
            float: p_max / P, dimensionless.

        Raises:
            ValueError: If no bushing material was given.
        """
        self._require_material()
        return self._bushing_data["p_max"] / self.pressure

    def velocity_safety_factor(self):
        """Allowable over actual rubbing velocity.

        Returns:
            float: v_max / V, dimensionless.

        Raises:
            ValueError: If no bushing material was given.
        """
        self._require_material()
        return self._bushing_data["v_max"] / self.rubbing_velocity

    def pv_safety_factor(self):
        """Allowable over actual PV product.

        Usually the binding limit of the three: a bushing can be within
        both its pressure and its velocity rating and still be over its
        PV rating.

        Returns:
            float: pv_max / PV, dimensionless.

        Raises:
            ValueError: If no bushing material was given.
        """
        self._require_material()
        return self._bushing_data["pv_max"] / self.pv

    def pv_check(self, temperature=None):
        """Evaluate every material limit at once.

        Args:
            temperature (float): Operating temperature in degrees Celsius
                (optional; the check is reported as None when absent).

        Returns:
            dict: ``pressure``, ``velocity``, ``pv`` and ``temperature``
            mapped to True/False, or None for a skipped check.

        Raises:
            ValueError: If no bushing material was given.
        """
        self._require_material()
        return {
            "pressure": self.pressure <= self._bushing_data["p_max"],
            "velocity": self.rubbing_velocity <= self._bushing_data["v_max"],
            "pv": self.pv <= self._bushing_data["pv_max"],
            "temperature": (
                None
                if temperature is None
                else to_magnitude(temperature, "degC") <= self._bushing_data["t_max"]
            ),
        }

    def maximum_speed(self):
        """Highest speed the PV and velocity limits allow, in rev/s.

        Returns:
            float: Limiting journal speed in rev/s.

        Raises:
            ValueError: If no bushing material was given.
        """
        self._require_material()
        velocity_limit = self._bushing_data["v_max"]
        pv_limit = self._bushing_data["pv_max"] / self.pressure
        return min(velocity_limit, pv_limit) * 1000.0 / (math.pi * self.bore_diameter)

    def maximum_load(self):
        """Highest load the pressure and PV limits allow, in N.

        Returns:
            float: Limiting radial load in N.

        Raises:
            ValueError: If no bushing material was given.
        """
        self._require_material()
        area = self.bore_diameter * self.length
        pressure_limit = self._bushing_data["p_max"]
        pv_limit = self._bushing_data["pv_max"] / self.rubbing_velocity
        return min(pressure_limit, pv_limit) * area

    # ---- Wear ----

    def wear_depth(self, hours, wear_factor):
        """Radial wear depth after a running time.

        ::

            w = k * P * V * t                (linear wear model)

        The wear factor must be supplied: it depends on the counterface
        material, hardness and surface finish as much as on the bushing,
        so :mod:`~mecapy.bearings.bushing_data` deliberately does not
        tabulate one.

        Args:
            hours (float): Running time in hours.
            wear_factor (float): Wear factor k in mm^3/(N*m), from test
                or supplier data.

        Returns:
            float: Radial wear depth in mm.

        Raises:
            ValueError: For a negative time or non-positive wear factor.
        """
        if hours < 0:
            raise ValueError("Running time must be non-negative")
        if wear_factor <= 0:
            raise ValueError("Wear factor must be strictly positive")
        # k [mm^3/(N*m)] * P [MPa = N/mm^2] * V [m/s] * t [s] -> mm
        return wear_factor * self.pressure * self.rubbing_velocity * hours * 3600.0

    def hours_to_wear(self, wear_limit, wear_factor):
        """Running time to reach a wear depth (inverse of :meth:`wear_depth`).

        Args:
            wear_limit (float): Allowable radial wear in mm.
            wear_factor (float): Wear factor k in mm^3/(N*m).

        Returns:
            float: Running time in hours.

        Raises:
            ValueError: For a non-positive limit or wear factor.
        """
        if wear_limit <= 0:
            raise ValueError("Wear limit must be strictly positive")
        if wear_factor <= 0:
            raise ValueError("Wear factor must be strictly positive")
        return wear_limit / (
            wear_factor * self.pressure * self.rubbing_velocity * 3600.0
        )

    # ---- Report ----

    def describe(self):
        """
        Human-readable summary of the bushing and its margins.

        The string is returned, not printed; use
        ``print(bearing.describe())``.

        Returns:
            str: Multi-line description, one quantity per line as
            ``label (symbol) = value unit``.
        """
        header = f"{self.__class__.__name__} geometry"
        if self.name:
            header += f" '{self.name}'"
        lines = [
            header,
            "=" * 40,
            f"bore diameter (d) = {self.bore_diameter:.3f} mm",
            f"length (l) = {self.length:.3f} mm",
            f"length ratio (l/d) = {self.l_over_d:.3f}",
            f"radial load (W) = {self.load:.1f} N",
            f"speed (N) = {self.speed:.3f} rev/s ({self.speed_rpm:.1f} rpm)",
            f"bearing pressure (P) = {self.pressure:.4f} MPa",
            f"rubbing velocity (V) = {self.rubbing_velocity:.4f} m/s",
            f"severity index (PV) = {self.pv:.4f} MPa*m/s",
            f"friction coefficient (mu) = {self.mu:.3f}",
            f"friction torque (T) = {self.friction_torque():.1f} N*mm",
            f"power loss (H) = {self.power_loss():.2f} W",
        ]
        if self._bushing_data is not None:
            lines += [
                f"liner = {self.bushing_material}",
                f"pressure safety = {self.pressure_safety_factor():.2f}",
                f"velocity safety = {self.velocity_safety_factor():.2f}",
                f"PV safety = {self.pv_safety_factor():.2f}",
                f"maximum load = {self.maximum_load():.1f} N",
                f"maximum speed = {self.maximum_speed():.2f} rev/s",
            ]
        else:
            lines.append("liner = not given (PV checks unavailable)")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"PlainBearing(d={self.bore_diameter}, l={self.length}, "
            f"W={self.load} N, N={self.speed} rev/s)"
        )
