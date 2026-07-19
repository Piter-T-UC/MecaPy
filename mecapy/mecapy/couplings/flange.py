"""Rigid flange coupling design and analysis module.

Units convention: geometry in mm, forces in N, torques in N*mm and
stresses in MPa, consistent with the other element modules.

Reference: classic rigid-coupling statics (Shigley Ch. 16 couplings
region and standard machine-design texts): the torque is shared by the
bolts on the bolt circle, and every load path — bolt shear, bolt
bearing on the flange, flange shear at the hub, and the shaft key — is
checked against yielding. Shear yield uses the maximum-shear-stress
criterion (0.577*Sy is the distortion-energy value; MSS uses 0.5*Sy —
here the distortion-energy 0.577*Sy is used, matching common practice).
"""

import math

from ..base import MechaElement
from ..materials import get_material_properties
from ..utils.constants import SAFETY_FACTOR_STATIC

SHEAR_YIELD_FACTOR = 0.577  # distortion-energy shear yield, Ssy = 0.577*Sy


class FlangeCoupling(MechaElement):
    """
    Rigid flange coupling — two bolted flanges joining coaxial shafts.

    Torque enters through the shaft key, flows through the hub and
    flange, and crosses the joint as shear in the bolts (assumed to
    share the load equally on the bolt circle). Each failure mode has a
    stress method and a safety factor; :meth:`torque_capacity` inverts
    them all and reports the weakest link.

    Attributes:
        shaft_diameter (float): Shaft diameter in mm.
        bolt_circle_diameter (float): Bolt-circle diameter Db in mm.
        n_bolts (int): Number of bolts.
        bolt_diameter (float): Bolt shank diameter in mm.
        flange_thickness (float): Flange (web) thickness in mm.
        hub_diameter (float): Hub outer diameter in mm, or None.
        key_width (float): Key width w in mm, or None.
        key_height (float): Key height h in mm, or None.
        key_length (float): Key length L in mm, or None.
        bolt_material (str): Bolt material type.
        material (str): Flange/hub material type.
    """

    def __init__(self, shaft_diameter, bolt_circle_diameter, n_bolts,
                 bolt_diameter, flange_thickness, hub_diameter=None,
                 key_width=None, key_height=None, key_length=None,
                 bolt_material="steel", material="cast_iron", name=None):
        """
        Initialize a FlangeCoupling object.

        Args:
            shaft_diameter (float): Shaft diameter in mm.
            bolt_circle_diameter (float): Bolt-circle diameter Db in mm.
                Must exceed the hub diameter (when given) and the shaft
                diameter.
            n_bolts (int): Number of bolts (>= 2).
            bolt_diameter (float): Bolt shank diameter in mm.
            flange_thickness (float): Flange thickness in mm.
            hub_diameter (float): Optional hub outer diameter in mm,
                needed for :meth:`hub_flange_shear_stress`. Must exceed
                the shaft diameter.
            key_width (float): Optional key width w in mm. The three key
                arguments must be given together or not at all.
            key_height (float): Optional key height h in mm.
            key_length (float): Optional key length L in mm.
            bolt_material (str): Bolt material type (default: "steel").
            material (str): Flange/hub material type (default: "cast_iron").
            name (str): Optional identifier for the coupling.

        Raises:
            ValueError: For non-positive dimensions, ``n_bolts`` below 2,
                inconsistent diameter ordering, or key arguments given
                only partially.
        """
        super().__init__(name=name, material=material)
        if shaft_diameter <= 0:
            raise ValueError("Shaft diameter must be strictly positive")
        if bolt_circle_diameter <= 0:
            raise ValueError("Bolt-circle diameter must be strictly positive")
        if bolt_diameter <= 0:
            raise ValueError("Bolt diameter must be strictly positive")
        if flange_thickness <= 0:
            raise ValueError("Flange thickness must be strictly positive")
        if not isinstance(n_bolts, int) or n_bolts < 2:
            raise ValueError("n_bolts must be an integer >= 2")
        if bolt_circle_diameter <= shaft_diameter:
            raise ValueError("Bolt-circle diameter must exceed the shaft diameter")
        if hub_diameter is not None:
            if hub_diameter <= shaft_diameter:
                raise ValueError("Hub diameter must exceed the shaft diameter")
            if bolt_circle_diameter <= hub_diameter:
                raise ValueError("Bolt-circle diameter must exceed the hub diameter")
        key_args = (key_width, key_height, key_length)
        if any(k is not None for k in key_args):
            if any(k is None for k in key_args):
                raise ValueError(
                    "key_width, key_height and key_length must be given together"
                )
            if any(k <= 0 for k in key_args):
                raise ValueError("Key dimensions must be strictly positive")

        self.shaft_diameter = shaft_diameter
        self.bolt_circle_diameter = bolt_circle_diameter
        self.n_bolts = n_bolts
        self.bolt_diameter = bolt_diameter
        self.flange_thickness = flange_thickness
        self.hub_diameter = hub_diameter
        self.key_width = key_width
        self.key_height = key_height
        self.key_length = key_length
        self.bolt_material = bolt_material

    # ---- Bolts ----

    @property
    def bolt_area(self):
        """float: Shank cross-section area of one bolt pi*d^2/4 in mm^2."""
        return math.pi * self.bolt_diameter ** 2 / 4

    def bolt_force(self, torque):
        """
        Tangential shear force carried by one bolt.

        Assumes equal sharing over the bolt circle::

            F = 2*T / (n * Db)

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Shear force per bolt in N.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        self._check_torque(torque)
        return 2 * torque / (self.n_bolts * self.bolt_circle_diameter)

    def bolt_shear_stress(self, torque):
        """
        Direct shear stress in one bolt shank.

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Shear stress in MPa.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        return self.bolt_force(torque) / self.bolt_area

    def bolt_safety_factor(self, torque):
        """
        Safety factor of the bolts against shear yielding.

        Compares the bolt shear stress with the distortion-energy shear
        yield ``0.577 * Sy`` of the bolt material (Pa -> MPa).

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Safety factor.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        sy = get_material_properties(self.bolt_material)["yield_strength"] / 1e6
        return SHEAR_YIELD_FACTOR * sy / self.bolt_shear_stress(torque)

    def flange_bearing_stress(self, torque):
        """
        Bearing (crushing) stress of one bolt on the flange hole.

        Projected bearing area is bolt diameter times flange thickness::

            sigma = F / (d_bolt * t_flange)

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Bearing stress in MPa.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        return self.bolt_force(torque) / (self.bolt_diameter * self.flange_thickness)

    def flange_bearing_safety_factor(self, torque):
        """
        Safety factor of the flange hole against bearing yield.

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Ratio of the flange yield strength to the bearing stress.
        """
        sy = self.material_properties["yield_strength"] / 1e6
        return sy / self.flange_bearing_stress(torque)

    # ---- Flange at the hub ----

    def hub_flange_shear_stress(self, torque):
        """
        Shear stress in the flange web at the hub outside diameter.

        The flange can tear in shear around the cylinder where it meets
        the hub::

            tau = 2*T / (pi * D_hub^2 * t_flange)

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Shear stress in MPa.

        Raises:
            ValueError: If ``hub_diameter`` was not given, or ``torque``
                is not strictly positive.
        """
        self._check_torque(torque)
        if self.hub_diameter is None:
            raise ValueError("hub_diameter is required for the hub-flange shear check")
        return 2 * torque / (math.pi * self.hub_diameter ** 2 * self.flange_thickness)

    def hub_flange_safety_factor(self, torque):
        """
        Safety factor of the flange web against shear yielding at the hub.

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Safety factor against 0.577*Sy of the flange material.
        """
        sy = self.material_properties["yield_strength"] / 1e6
        return SHEAR_YIELD_FACTOR * sy / self.hub_flange_shear_stress(torque)

    # ---- Key ----

    def key_force(self, torque):
        """
        Tangential force on the shaft key.

        ::

            F_k = 2*T / d_shaft

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Key force in N.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        self._check_torque(torque)
        return 2 * torque / self.shaft_diameter

    def key_shear_stress(self, torque):
        """
        Shear stress across the key section.

        ::

            tau = F_k / (w * L)

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Key shear stress in MPa.

        Raises:
            ValueError: If the key dimensions were not given, or
                ``torque`` is not strictly positive.
        """
        self._require_key()
        return self.key_force(torque) / (self.key_width * self.key_length)

    def key_bearing_stress(self, torque):
        """
        Bearing (crushing) stress on the key flank.

        The force bears on half the key height::

            sigma = 2*F_k / (h * L)

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Key bearing stress in MPa.

        Raises:
            ValueError: If the key dimensions were not given, or
                ``torque`` is not strictly positive.
        """
        self._require_key()
        return 2 * self.key_force(torque) / (self.key_height * self.key_length)

    def key_safety_factor(self, torque):
        """
        Safety factor of the key: the worse of shear and bearing.

        Shear is checked against ``0.577 * Sy`` and bearing against
        ``Sy`` of the coupling material (a conservative stand-in for the
        key steel when they differ).

        Args:
            torque (float): Transmitted torque in N*mm.

        Returns:
            float: Minimum of the shear and bearing safety factors.

        Raises:
            ValueError: If the key dimensions were not given, or
                ``torque`` is not strictly positive.
        """
        sy = self.material_properties["yield_strength"] / 1e6
        sf_shear = SHEAR_YIELD_FACTOR * sy / self.key_shear_stress(torque)
        sf_bearing = sy / self.key_bearing_stress(torque)
        return min(sf_shear, sf_bearing)

    # ---- Overall capacity ----

    def torque_capacity(self, safety_factor=SAFETY_FACTOR_STATIC):
        """
        Largest torque the coupling carries at a given safety factor.

        Inverts every available failure mode (bolt shear, flange
        bearing, hub-flange shear and key checks — the last two only
        when their geometry was given; all stresses are linear in the
        torque) and returns the smallest governing torque.

        Args:
            safety_factor (float): Required safety factor (default:
                the static factor from :mod:`mecapy.utils.constants`).

        Returns:
            float: Allowable torque in N*mm.

        Raises:
            ValueError: If ``safety_factor`` is not strictly positive.
        """
        if safety_factor <= 0:
            raise ValueError("Safety factor must be strictly positive")
        reference_torque = 1.0e6  # any torque works: stresses are linear in T
        factors = [self.bolt_safety_factor(reference_torque),
                   self.flange_bearing_safety_factor(reference_torque)]
        if self.hub_diameter is not None:
            factors.append(self.hub_flange_safety_factor(reference_torque))
        if self.key_width is not None:
            factors.append(self.key_safety_factor(reference_torque))
        return reference_torque * min(factors) / safety_factor

    # ---- Internal helpers ----

    @staticmethod
    def _check_torque(torque):
        """Validate a transmitted torque."""
        if torque <= 0:
            raise ValueError("Torque must be strictly positive")

    def _require_key(self):
        """Raise if the key geometry was not given."""
        if self.key_width is None:
            raise ValueError(
                "Key dimensions (key_width, key_height, key_length) are "
                "required for the key checks"
            )
