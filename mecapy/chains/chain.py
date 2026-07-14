"""Roller chain drive design and analysis module.

Implements standard roller-chain geometry and kinematics. Lengths use a
consistent unit (the chain pitch ``p``); speeds are in rev/min and the
resulting chain velocity is in the same length unit per second.
"""

import math

from ..base import MechaElement


class Chain(MechaElement):
    """
    Roller chain drive.

    Provides sprocket pitch diameter, chain length, chain velocity and the
    chordal-speed variation inherent to chain drives. Inherits from
    :class:`~mecapy.base.MechaElement`.

    Attributes:
        pitch (float): Chain pitch p (length units, e.g. mm).
        teeth (int): Number of teeth on the driving sprocket.
        strands (int): Number of parallel strands.
        material (str): Chain material.
    """

    def __init__(self, pitch, teeth, strands=1, material="steel", name=None):
        """
        Initialize a Chain object.

        Args:
            pitch (float): Chain pitch p (length units, e.g. mm).
            teeth (int): Number of teeth on the driving sprocket.
            strands (int): Number of parallel strands (default: 1).
            material (str): Chain material (default: "steel").
            name (str): Optional identifier for the chain.
        """
        super().__init__(name=name, material=material)
        self.pitch = pitch
        self.teeth = teeth
        self.strands = strands

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def pitch_diameter(self, teeth=None):
        """
        Sprocket pitch diameter ``D = p / sin(pi / N)``.

        Args:
            teeth (int): Number of teeth. Defaults to this chain's driving
                sprocket teeth count.

        Returns:
            float: Pitch diameter in the same length units as the pitch.
        """
        N = teeth if teeth is not None else self.teeth
        return self.pitch / math.sin(math.pi / N)

    def length_in_pitches(self, driven_teeth, center_distance):
        """
        Chain length in pitches for a two-sprocket drive.

        ``L = 2C + (N1 + N2)/2 + (N2 - N1)^2 / (4 * pi^2 * C)``

        Args:
            driven_teeth (int): Teeth on the driven sprocket N2.
            center_distance (float): Center distance expressed in pitches.

        Returns:
            float: Chain length in pitches (typically rounded up to an even
            integer in practice).
        """
        N1 = self.teeth
        N2 = driven_teeth
        C = center_distance
        return 2 * C + (N1 + N2) / 2 + (N2 - N1) ** 2 / (4 * math.pi ** 2 * C)

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    def velocity(self, speed):
        """
        Mean chain velocity.

        ``v = N * p * n / 60`` (per second) with ``n`` in rev/min.

        Args:
            speed (float): Driving sprocket speed in rev/min.

        Returns:
            float: Chain velocity in length-units per second.
        """
        return self.teeth * self.pitch * speed / 60.0

    def chordal_speed_variation(self):
        """
        Fractional chordal (polygonal) speed variation of the chain.

        ``(v_max - v_min) / v_mean = (pi / N) * (1 / sin(pi/N) - 1/tan(pi/N))``

        Returns:
            float: Chordal speed variation as a fraction (multiply by 100
            for a percentage).
        """
        angle = math.pi / self.teeth
        return angle * (1 / math.sin(angle) - 1 / math.tan(angle))

    def __repr__(self):
        return f"Chain(pitch={self.pitch}, teeth={self.teeth}, strands={self.strands})"
