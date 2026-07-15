"""Planetary (epicyclic) gear sets.

A simple planetary set has a central sun gear, several planet gears on a
rotating carrier, and an internal ring gear. Kinematics follow the
Willis train-value equation

    (w_sun - w_carrier) / (w_ring - w_carrier) = -Zr / Zs

Fixing one member (sun, ring or carrier) gives six possible speed-ratio
configurations.
"""

import math

from .gear import Gear


_MEMBERS = ("sun", "ring", "carrier")


class PlanetaryGearSet:
    """
    Simple planetary (epicyclic) gear set.

    Accepts gear objects (e.g. :class:`~mecapy.gears.SpurGear`) or plain
    integer teeth counts; with plain integers only kinematics are
    available (no geometry checks between members beyond teeth counts).

    Attributes:
        sun (Gear or int): Sun gear (external, central).
        planet (Gear or int): One planet gear (all planets identical).
        ring (Gear or int): Ring gear (internal teeth).
        n_planets (int): Number of planets (default: 3).
        name (str): Optional identifier.
    """

    def __init__(self, sun, planet, ring, n_planets=3, name=None):
        """
        Initialize a planetary gear set and validate its assembly.

        Args:
            sun (Gear or int): Sun gear or its teeth count.
            planet (Gear or int): Planet gear or its teeth count.
            ring (Gear or int): Ring gear or its teeth count.
            n_planets (int): Number of equally spaced planets (>= 1).
            name (str): Optional identifier.

        Raises:
            ValueError: If the geometric condition ``Zr = Zs + 2 Zp``
                fails, the assembly condition ``(Zs + Zr) % n_planets``
                fails, adjacent planets would collide, or (for gear
                objects) modules/pressure angles do not match.
        """
        self.sun = sun
        self.planet = planet
        self.ring = ring
        self.name = name
        if n_planets != int(n_planets) or n_planets < 1:
            raise ValueError("Number of planets must be an integer >= 1")
        self.n_planets = int(n_planets)

        zs, zp, zr = self.sun_teeth, self.planet_teeth, self.ring_teeth
        # Geometric (concentricity) condition.
        if zr != zs + 2 * zp:
            raise ValueError(
                f"Geometric condition failed: ring teeth {zr} != "
                f"sun {zs} + 2 * planet {zp} = {zs + 2 * zp}"
            )
        # Assembly condition for equally spaced planets.
        if (zs + zr) % self.n_planets != 0:
            raise ValueError(
                f"Assembly condition failed: (sun {zs} + ring {zr}) is not "
                f"divisible by {self.n_planets} planets"
            )
        # Consistency between gear objects.
        gears = [g for g in (sun, planet, ring) if isinstance(g, Gear)]
        if gears:
            m0, pa0 = gears[0].module, gears[0].pressure_angle
            for g in gears[1:]:
                if not math.isclose(g.module, m0):
                    raise ValueError(
                        "All planetary members must share the same module"
                    )
                if not math.isclose(g.pressure_angle, pa0):
                    raise ValueError(
                        "All planetary members must share the same "
                        "pressure angle"
                    )
            # Adjacency: tip circles of neighbouring planets must clear.
            if self.n_planets >= 2:
                m = m0
                carrier_radius = m * (zs + zp) / 2
                planet_tip_diameter = m * zp + 2 * m  # d + 2 * addendum
                clearance = (2 * carrier_radius
                             * math.sin(math.pi / self.n_planets))
                if clearance <= planet_tip_diameter:
                    raise ValueError(
                        f"Adjacent planets collide: spacing {clearance:.2f} mm"
                        f" <= planet tip diameter {planet_tip_diameter:.2f} mm"
                    )

    # ------------------------------------------------------------------
    # Teeth counts
    # ------------------------------------------------------------------

    @staticmethod
    def _teeth(member):
        return member.teeth if isinstance(member, Gear) else int(member)

    @property
    def sun_teeth(self):
        """int: Sun gear teeth count."""
        return self._teeth(self.sun)

    @property
    def planet_teeth(self):
        """int: Planet gear teeth count."""
        return self._teeth(self.planet)

    @property
    def ring_teeth(self):
        """int: Ring gear teeth count."""
        return self._teeth(self.ring)

    # ------------------------------------------------------------------
    # Kinematics (Willis equation)
    # ------------------------------------------------------------------

    def _solve_speeds(self, input_member, input_speed, fixed):
        """Return {sun, ring, carrier} speeds from the Willis equation."""
        if input_member not in _MEMBERS or fixed not in _MEMBERS:
            raise ValueError(f"Members must be one of {_MEMBERS}")
        if input_member == fixed:
            raise ValueError("Input member cannot be the fixed member")
        zs, zr = self.sun_teeth, self.ring_teeth
        # Willis: (w_sun - w_c) = -Zr/Zs * (w_ring - w_c)
        speeds = {m: None for m in _MEMBERS}
        speeds[fixed] = 0.0
        speeds[input_member] = float(input_speed)
        # Solve for the remaining member from
        # w_sun - w_c + (Zr/Zs) * (w_ring - w_c) = 0
        # -> Zs*w_sun + Zr*w_ring - (Zs+Zr)*w_c = 0
        unknown = next(m for m in _MEMBERS if speeds[m] is None)
        ws, wr, wc = speeds["sun"], speeds["ring"], speeds["carrier"]
        if unknown == "sun":
            speeds["sun"] = ((zs + zr) * wc - zr * wr) / zs
        elif unknown == "ring":
            speeds["ring"] = ((zs + zr) * wc - zs * ws) / zr
        else:
            speeds["carrier"] = (zs * ws + zr * wr) / (zs + zr)
        return speeds

    def ratio(self, input="sun", output="carrier", fixed="ring"):
        """
        Speed ratio ``w_input / w_output`` for a given configuration.

        The six configurations come from choosing distinct input, output
        and fixed members among sun, ring and carrier. For example, with
        the ring fixed and sun driving the carrier the ratio is
        ``1 + Zr/Zs``.

        Args:
            input (str): Input member ("sun", "ring" or "carrier").
            output (str): Output member.
            fixed (str): Fixed (grounded) member.

        Returns:
            float: Speed ratio (negative means the output rotates
            opposite to the input).

        Raises:
            ValueError: If the three members are not distinct or invalid.
        """
        if len({input, output, fixed}) != 3:
            raise ValueError("Input, output and fixed members must be distinct")
        speeds = self._solve_speeds(input, 1.0, fixed)
        if speeds[output] == 0:
            raise ValueError("Output member does not rotate in this configuration")
        return 1.0 / speeds[output]

    def speeds(self, input_member, input_speed_rpm, fixed):
        """
        All member speeds for a given input speed.

        Args:
            input_member (str): Driven-in member ("sun", "ring" or
                "carrier").
            input_speed_rpm (float): Input speed in rpm.
            fixed (str): Fixed member.

        Returns:
            dict: Speeds in rpm for "sun", "ring", "carrier" and
            "planet" (planet spin about its own axis, from the
            sun-planet mesh relative to the carrier).
        """
        speeds = self._solve_speeds(input_member, input_speed_rpm, fixed)
        zs, zp = self.sun_teeth, self.planet_teeth
        # Planet spin relative to carrier from the sun-planet external mesh.
        speeds["planet"] = (speeds["carrier"]
                            - zs / zp * (speeds["sun"] - speeds["carrier"]))
        return speeds

    def torques(self, input_member, input_torque, fixed):
        """
        Static torque balance for a given input torque (ideal, no losses).

        Member torques are in proportion ``Ts : Tr : Tc =
        Zs : Zr : -(Zs + Zr)``, scaled so the input member carries
        ``input_torque``. The carrier is not a gear; its torque reacts
        the sum of the other two.

        Args:
            input_member (str): Member where the torque is applied.
            input_torque (float): Applied torque (any consistent unit).
            fixed (str): Fixed member (carries a reaction torque).

        Returns:
            dict: Torques on "sun", "ring" and "carrier" in the same
            unit as ``input_torque``.

        Raises:
            ValueError: If members are invalid or not distinct.
        """
        if input_member not in _MEMBERS or fixed not in _MEMBERS:
            raise ValueError(f"Members must be one of {_MEMBERS}")
        if input_member == fixed:
            raise ValueError("Input member cannot be the fixed member")
        zs, zr = self.sun_teeth, self.ring_teeth
        proportions = {"sun": zs, "ring": zr, "carrier": -(zs + zr)}
        scale = input_torque / proportions[input_member]
        return {m: proportions[m] * scale for m in _MEMBERS}

    def __repr__(self):
        return (
            f"PlanetaryGearSet(sun={self.sun_teeth}, "
            f"planet={self.planet_teeth}, ring={self.ring_teeth}, "
            f"n_planets={self.n_planets})"
        )
