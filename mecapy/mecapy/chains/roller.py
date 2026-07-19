"""Roller chain design and analysis module.

Units convention: geometry in mm, forces in N, chain speed in m/s and
power in W (converted once at the boundary in :meth:`RollerChain.rated_power`,
whose underlying empirical equations are inch/hp-native).

Reference: Shigley's *Mechanical Engineering Design*, Ch. 17
(Sec. 17-4, roller chain, Eqs. 17-28 to 17-40).
"""

import math

from ..base import MechaElement
from ..utils.converters import hp_to_kw
from .chain_data import STRAND_FACTORS, get_chain


class RollerChain(MechaElement):
    """
    ANSI roller chain drive between two sprockets.

    Attributes:
        chain_number (str): ANSI chain number.
        driver_teeth (int): Number of teeth on the driver sprocket, N1.
        driven_teeth (int): Number of teeth on the driven sprocket, N2.
        center_distance (float): Center-to-center distance in mm, or None.
        strands (int): Number of chain strands.
    """

    def __init__(self, chain_number, driver_teeth, driven_teeth,
                 center_distance=None, strands=1, material="steel", name=None):
        """
        Initialize a RollerChain object.

        Args:
            chain_number (int or str): ANSI chain number, a key of
                :data:`mecapy.chains.chain_data.ANSI_ROLLER_CHAINS`.
            driver_teeth (int): Number of teeth on the driver sprocket,
                N1 (integer >= 3).
            driven_teeth (int): Number of teeth on the driven sprocket,
                N2 (integer >= 3).
            center_distance (float): Optional center-to-center distance in
                mm. Methods that need it (:meth:`chain_length_pitches`)
                accept an explicit override argument when it was not given.
            strands (int): Number of chain strands, a key of
                :data:`mecapy.chains.chain_data.STRAND_FACTORS` (default 1).
            material (str): Structural material type (default: "steel").
            name (str): Optional identifier for the chain.

        Raises:
            ValueError: For an unknown ``chain_number``, non-integer or
                too-small tooth counts, an unsupported ``strands`` count,
                or a non-positive ``center_distance``.
        """
        super().__init__(name=name, material=material)
        get_chain(chain_number)
        if not isinstance(driver_teeth, int) or driver_teeth < 3:
            raise ValueError("driver_teeth must be an integer >= 3")
        if not isinstance(driven_teeth, int) or driven_teeth < 3:
            raise ValueError("driven_teeth must be an integer >= 3")
        if strands not in STRAND_FACTORS:
            raise ValueError(f"strands must be one of {sorted(STRAND_FACTORS)}")
        if center_distance is not None and center_distance <= 0:
            raise ValueError("center_distance must be strictly positive")

        self.chain_number = str(chain_number)
        self.driver_teeth = driver_teeth
        self.driven_teeth = driven_teeth
        self.center_distance = center_distance
        self.strands = strands

    # ---- Geometry ----

    @property
    def pitch(self):
        """float: Chain pitch p in mm."""
        return get_chain(self.chain_number)["pitch"]

    @property
    def speed_ratio(self):
        """float: Driven/driver speed ratio = driver_teeth / driven_teeth."""
        return self.driver_teeth / self.driven_teeth

    @property
    def driver_pitch_diameter(self):
        """float: Driver sprocket pitch diameter in mm (Eq. 17-28)."""
        return self.pitch / math.sin(math.radians(180.0 / self.driver_teeth))

    @property
    def driven_pitch_diameter(self):
        """float: Driven sprocket pitch diameter in mm (Eq. 17-28)."""
        return self.pitch / math.sin(math.radians(180.0 / self.driven_teeth))

    @property
    def meets_recommended_teeth(self):
        """bool: True if the driver sprocket meets Shigley's >= 17 tooth recommendation."""
        return self.driver_teeth >= 17

    @property
    def chordal_speed_variation(self):
        """float: Fractional chordal speed variation at the driver sprocket."""
        half_angle = math.pi / self.driver_teeth
        return half_angle * (1 / math.sin(half_angle) - 1 / math.tan(half_angle))

    def chain_length_pitches(self, center_distance=None):
        """
        Chain length in pitches for a given center distance (Eq. 17-34).

        Args:
            center_distance (float): Center-to-center distance in mm.
                Defaults to the value given at construction.

        Returns:
            float: Chain length L/p in pitches.

        Raises:
            ValueError: If no center distance is available (neither given
                here nor at construction), or it is not strictly positive.
        """
        c = center_distance if center_distance is not None else self.center_distance
        if c is None:
            raise ValueError("center_distance is required: pass one, or set it at construction")
        if c <= 0:
            raise ValueError("center_distance must be strictly positive")
        p = self.pitch
        n1, n2 = self.driver_teeth, self.driven_teeth
        return (2 * c / p + (n1 + n2) / 2
                + ((n2 - n1) / (2 * math.pi)) ** 2 * (p / c))

    def chain_length(self, center_distance=None):
        """
        Chain length in mm for a given center distance.

        Args:
            center_distance (float): See :meth:`chain_length_pitches`.

        Returns:
            float: Chain length in mm.

        Raises:
            ValueError: See :meth:`chain_length_pitches`.
        """
        return self.chain_length_pitches(center_distance) * self.pitch

    def center_distance_for_length(self, length_pitches):
        """
        Center distance that gives a target chain length.

        Inverts :meth:`chain_length_pitches` (Eq. 17-35/17-36) -- the
        round-trip partner.

        Args:
            length_pitches (float): Target chain length in pitches, L/p.

        Returns:
            float: Center distance C in mm.

        Raises:
            ValueError: If ``length_pitches`` is not strictly positive, or
                is too short to span the two sprockets.
        """
        if length_pitches <= 0:
            raise ValueError("length_pitches must be strictly positive")
        p = self.pitch
        n1, n2 = self.driver_teeth, self.driven_teeth
        a = (n1 + n2) / 2 - length_pitches
        discriminant = a ** 2 - 8 * ((n2 - n1) / (2 * math.pi)) ** 2
        if discriminant < 0:
            raise ValueError("length_pitches is too short to span the two sprockets")
        return (p / 4) * (-a + math.sqrt(discriminant))

    # ---- Kinematics and strength ----

    def chain_speed(self, rpm):
        """
        Chain (pitch-line) speed at the driver sprocket.

        Args:
            rpm (float): Driver sprocket speed in rev/min.

        Returns:
            float: Chain speed in m/s.

        Raises:
            ValueError: If ``rpm`` is not strictly positive.
        """
        if rpm <= 0:
            raise ValueError("Speed must be strictly positive")
        return self.driver_teeth * self.pitch * rpm / 60000.0

    def working_tension(self, power, rpm):
        """
        Working chain tension for a given transmitted power.

        Args:
            power (float): Transmitted power in W.
            rpm (float): Driver sprocket speed in rev/min.

        Returns:
            float: Working tension in N.

        Raises:
            ValueError: If ``power`` is not strictly positive; see
                :meth:`chain_speed`.
        """
        if power <= 0:
            raise ValueError("Power must be strictly positive")
        return power / self.chain_speed(rpm)

    def tensile_safety_factor(self, power, rpm):
        """
        Safety factor against the chain's minimum tensile strength.

        Args:
            power (float): Transmitted power in W.
            rpm (float): Driver sprocket speed in rev/min.

        Returns:
            float: strands * min_tensile_strength / working_tension.

        Raises:
            ValueError: See :meth:`working_tension`.
        """
        breaking_load = get_chain(self.chain_number)["min_tensile_strength"]
        return self.strands * breaking_load / self.working_tension(power, rpm)

    def rated_power(self, rpm):
        """
        Rated power at a given driver speed, min(H1, H2).

        Uses the empirical pre-extreme-horsepower equations (Eq. 17-32,
        link-plate fatigue; Eq. 17-33, roller-bushing wear), which are
        inch/hp-native; computed internally in those units and converted
        once at the boundary. Service factors Ks and lubrication regime
        selection are not modeled.

        Args:
            rpm (float): Driver sprocket speed in rev/min.

        Returns:
            float: Rated power in W, for :attr:`strands` chains.

        Raises:
            ValueError: If ``rpm`` is not strictly positive.
        """
        if rpm <= 0:
            raise ValueError("Speed must be strictly positive")
        chain = get_chain(self.chain_number)
        p_in = self.pitch / 25.4
        n1 = self.driver_teeth
        h1 = 0.004 * n1 ** 1.08 * rpm ** 0.9 * p_in ** (3 - 0.07 * p_in)
        h2 = 1000 * chain["kr"] * n1 ** 1.5 * p_in ** 0.8 / rpm ** 1.5
        h_single = min(h1, h2)
        h_total = h_single * STRAND_FACTORS[self.strands]
        return hp_to_kw(h_total) * 1000.0

    def power_safety_factor(self, power, rpm):
        """
        Ratio of the rated power to a given transmitted power.

        Args:
            power (float): Transmitted power in W.
            rpm (float): Driver sprocket speed in rev/min.

        Returns:
            float: rated_power(rpm) / power.

        Raises:
            ValueError: If ``power`` is not strictly positive; see
                :meth:`rated_power`.
        """
        if power <= 0:
            raise ValueError("Power must be strictly positive")
        return self.rated_power(rpm) / power
