"""V-belt design and analysis module.

Units convention: same as :mod:`mecapy.belts.flat` (mm, N, W, kg/m, MPa).

Reference: Shigley's *Mechanical Engineering Design*, Ch. 17
(Sec. 17-3, V-belts, Eqs. 17-16 onward). The wedging action of the belt in
the sheave groove multiplies the effective friction the same way a cone
clutch's half-angle does (:class:`mecapy.clutches.cone.ConeClutch`);
:data:`~mecapy.belts.belt_data.V_BELT_EFFECTIVE_MU` = 0.5123 is Shigley's
standard groove-effective friction value for the standard groove angle
(Eq. 17-24), confirmed throughout the Ch. 17 solutions manual.
"""

import math

from .belt_data import V_BELT_EFFECTIVE_MU, get_v_belt_section
from .flat import FlatBelt


class VBelt(FlatBelt):
    """
    V-belt — wedges into a grooved sheave, amplifying the effective friction.

    Subclasses :class:`FlatBelt`: the whole capstan core (wrap angles,
    belt speed, centrifugal tension, tension/power round-trip, initial
    tension) is inherited unchanged; only the effective friction and the
    section's tabulated mass per length are overridden.

    Attributes:
        section (str): V-belt section letter ("A".."E").
        groove_angle (float): Full sheave groove angle in degrees.
        n_belts (int): Number of belts running in parallel.
    """

    def __init__(self, section, driver_diameter, driven_diameter,
                 center_distance, groove_angle=36.0, mu=None, n_belts=1,
                 material="steel", name=None):
        """
        Initialize a VBelt object.

        Args:
            section (str): V-belt section letter, one of the keys of
                :data:`~mecapy.belts.belt_data.V_BELT_SECTIONS`.
            driver_diameter (float): Driver sheave pitch diameter in mm.
            driven_diameter (float): Driven sheave pitch diameter in mm.
                Both diameters must be at least the section's minimum
                sheave diameter.
            center_distance (float): Center-to-center distance C in mm.
            groove_angle (float): Full sheave groove angle in degrees, in
                (20, 60) (default 36, the standard V-belt groove angle).
            mu (float): Explicit belt/sheave friction coefficient. When
                given, the effective friction is ``mu / sin(groove_angle/2)``
                (the wedging amplification); when None (default), the
                effective friction is Shigley's tabulated
                ``V_BELT_EFFECTIVE_MU`` directly.
            n_belts (int): Number of belts running in parallel (>= 1).
            material (str): Structural material type (default: "steel").
            name (str): Optional identifier for the belt.

        Raises:
            ValueError: For an unknown ``section``, either sheave diameter
                below the section's minimum, a ``groove_angle`` outside
                (20, 60) degrees, or ``n_belts`` below 1; plus the checks
                in :class:`FlatBelt` (drive is always "open").
        """
        if not 20 < groove_angle < 60:
            raise ValueError("Groove angle must be in (20, 60) degrees")
        if not isinstance(n_belts, int) or n_belts < 1:
            raise ValueError("n_belts must be an integer >= 1")
        section_data = get_v_belt_section(section)
        if min(driver_diameter, driven_diameter) < section_data["min_sheave_diameter"]:
            raise ValueError(
                f"Sheave diameter is below the {section} section minimum "
                f"of {section_data['min_sheave_diameter']} mm")

        base_mu = mu if mu is not None else V_BELT_EFFECTIVE_MU
        super().__init__(width=section_data["width"], thickness=section_data["thickness"],
                         driver_diameter=driver_diameter, driven_diameter=driven_diameter,
                         center_distance=center_distance, mu=base_mu, drive="open",
                         material=material, name=name)
        self.section = section.upper() if isinstance(section, str) else section
        self.groove_angle = groove_angle
        self.n_belts = n_belts
        self._explicit_mu = mu

    @property
    def effective_friction(self):
        """float: Groove-wedged effective friction (overrides FlatBelt's raw mu)."""
        if self._explicit_mu is not None:
            return self._explicit_mu / math.sin(math.radians(self.groove_angle / 2))
        return V_BELT_EFFECTIVE_MU

    @property
    def mass_per_length(self):
        """float: Tabulated section mass per unit length in kg/m (overrides FlatBelt)."""
        return get_v_belt_section(self.section)["mass_per_length"]

    @property
    def pitch_length(self):
        """float: Approximate pitch length Lp in mm (Eq. 17-16a)."""
        driven_d, driver_d, c = self.driven_diameter, self.driver_diameter, self.center_distance
        return (2 * c + math.pi * (driven_d + driver_d) / 2
                + (driven_d - driver_d) ** 2 / (4 * c))

    def center_distance_for_pitch_length(self, pitch_length):
        """
        Center distance that gives a target pitch length.

        Inverts :attr:`pitch_length` (Eq. 17-16b) -- the round-trip partner.

        Args:
            pitch_length (float): Target pitch length Lp in mm.

        Returns:
            float: Center distance C in mm.

        Raises:
            ValueError: If ``pitch_length`` is not strictly positive, or is
                too short to span the two sheaves.
        """
        if pitch_length <= 0:
            raise ValueError("Pitch length must be strictly positive")
        driven_d, driver_d = self.driven_diameter, self.driver_diameter
        x = pitch_length - math.pi * (driven_d + driver_d) / 2
        discriminant = x ** 2 - 2 * (driven_d - driver_d) ** 2
        if discriminant < 0:
            raise ValueError("Pitch length is too short to span the two sheaves")
        return 0.25 * (x + math.sqrt(discriminant))

    @property
    def designation(self):
        """str: Catalog designation, e.g. "B90" (section + inside length in inches)."""
        section_data = get_v_belt_section(self.section)
        inside_length_in = (self.pitch_length - section_data["length_addend"]) / 25.4
        return f"{self.section}{round(inside_length_in)}"

    def belts_required(self, design_power, per_belt_power):
        """
        Number of belts needed to carry a design power.

        Per-belt rating tables (Kb/Kc/K1/K2 corrections, Shigley Tables
        17-12..17-14) are not implemented; the caller supplies the rated
        power of one belt at the operating speed.

        Args:
            design_power (float): Total design power in W.
            per_belt_power (float): Rated power of one belt in W.

        Returns:
            int: Number of belts, rounded up.

        Raises:
            ValueError: If either argument is not strictly positive.
        """
        if design_power <= 0:
            raise ValueError("Design power must be strictly positive")
        if per_belt_power <= 0:
            raise ValueError("Per-belt power must be strictly positive")
        return math.ceil(design_power / per_belt_power)
