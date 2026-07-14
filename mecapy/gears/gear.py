"""Gear design and analysis module (AGMA method)."""

import math

from ..base import MechaElement
from . import agma


class Gear(MechaElement):
    """
    Spur gear design and analysis using the AGMA method.

    Provides AGMA bending-stress and contact-stress (pitting) calculations
    following Shigley's SI formulation. Inherits shared material and stress
    behaviour from :class:`~mecapy.base.MechaElement`.

    Units: module in mm, face width in mm, power in W, speed in rev/min,
    forces in N, velocities in m/s and stresses in MPa.

    Attributes:
        teeth (int): Number of teeth.
        module (float): Gear module in mm.
        face_width (float): Face width b in mm.
        pressure_angle (float): Pressure angle in degrees.
        quality_number (float): AGMA transmission accuracy grade Qv.
        material (str): Material type.
    """

    def __init__(self, teeth, module, face_width=None, pressure_angle=20.0,
                 quality_number=6, material="steel", name=None):
        """
        Initialize a Gear object.

        Args:
            teeth (int): Number of teeth.
            module (float): Gear module in mm.
            face_width (float): Face width b in mm. Defaults to ``10 * module``
                (a common rule of thumb) when not supplied.
            pressure_angle (float): Pressure angle in degrees (default: 20).
            quality_number (float): AGMA accuracy grade Qv (default: 6).
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the gear.
        """
        super().__init__(name=name, material=material)
        self.teeth = teeth
        self.module = module
        self.face_width = face_width if face_width is not None else 10 * module
        self.pressure_angle = pressure_angle
        self.quality_number = quality_number

    # ------------------------------------------------------------------
    # Geometry / kinematics
    # ------------------------------------------------------------------
    @property
    def pitch_diameter(self):
        """float: Pitch diameter d = module * teeth (mm)."""
        return self.module * self.teeth

    def pitch_line_velocity(self, speed):
        """
        Pitch-line velocity.

        Args:
            speed (float): Rotational speed in rev/min.

        Returns:
            float: Pitch-line velocity in m/s.
        """
        return math.pi * self.pitch_diameter * speed / 60000.0

    def tangential_load(self, power, speed):
        """
        Transmitted tangential load Wt.

        Args:
            power (float): Transmitted power in W.
            speed (float): Rotational speed in rev/min.

        Returns:
            float: Tangential load Wt in N.
        """
        velocity = self.pitch_line_velocity(speed)
        if velocity == 0:
            raise ValueError("Pitch-line velocity is zero; check the speed")
        return power / velocity

    def dynamic_factor(self, speed):
        """AGMA dynamic factor Kv at a given speed (rev/min)."""
        return agma.dynamic_factor(self.quality_number, self.pitch_line_velocity(speed))

    # ------------------------------------------------------------------
    # AGMA stresses
    # ------------------------------------------------------------------
    def bending_stress(self, power, speed, geometry_factor,
                       overload_factor=1.0, size_factor=1.0,
                       load_distribution_factor=1.0, rim_factor=1.0):
        """
        AGMA bending stress (Shigley SI form).

        ``sigma = Wt * Ko * Kv * Ks * (1 / (b * m)) * (KH * KB / YJ)``

        Args:
            power (float): Transmitted power in W.
            speed (float): Rotational speed in rev/min.
            geometry_factor (float): Bending geometry factor YJ (from AGMA
                charts, typically ~0.3-0.45 for spur gears).
            overload_factor (float): Overload factor Ko (default: 1).
            size_factor (float): Size factor Ks (default: 1).
            load_distribution_factor (float): Load-distribution factor KH
                (default: 1).
            rim_factor (float): Rim-thickness factor KB (default: 1).

        Returns:
            float: Bending stress in MPa.
        """
        Wt = self.tangential_load(power, speed)
        Kv = self.dynamic_factor(speed)
        return (
            Wt
            * overload_factor
            * Kv
            * size_factor
            * (1.0 / (self.face_width * self.module))
            * (load_distribution_factor * rim_factor / geometry_factor)
        )

    def contact_stress(self, power, speed, gear_ratio, mate=None,
                       overload_factor=1.0, size_factor=1.0,
                       load_distribution_factor=1.0, surface_factor=1.0):
        """
        AGMA contact (pitting) stress (Shigley SI form).

        ``sigma_c = ZE * sqrt(Wt * Ko * Kv * Ks * (KH / (d * b)) * (ZR / ZI))``

        Args:
            power (float): Transmitted power in W.
            speed (float): Rotational speed in rev/min.
            gear_ratio (float): Gear ratio mG = NG / NP (>= 1).
            mate (Gear): Meshing gear, used for the elastic coefficient. If
                ``None``, the mating gear is assumed to share this material.
            overload_factor (float): Overload factor Ko (default: 1).
            size_factor (float): Size factor Ks (default: 1).
            load_distribution_factor (float): Load-distribution factor KH
                (default: 1).
            surface_factor (float): Surface-condition factor ZR (default: 1).

        Returns:
            float: Contact stress in MPa.
        """
        Wt = self.tangential_load(power, speed)
        Kv = self.dynamic_factor(speed)
        props = self.material_properties
        mate_props = mate.material_properties if mate is not None else props
        Ze = agma.elastic_coefficient(
            props["elastic_modulus"], props["poisson_ratio"],
            mate_props["elastic_modulus"], mate_props["poisson_ratio"],
        )
        Zi = agma.pitting_geometry_factor(self.pressure_angle, gear_ratio)
        radical = (
            Wt
            * overload_factor
            * Kv
            * size_factor
            * (load_distribution_factor / (self.pitch_diameter * self.face_width))
            * (surface_factor / Zi)
        )
        return Ze * math.sqrt(radical)

    # ------------------------------------------------------------------
    # Safety factors
    # ------------------------------------------------------------------
    def bending_safety_factor(self, bending_stress, allowable_stress=None):
        """
        Safety factor against bending fatigue.

        Args:
            bending_stress (float): Applied AGMA bending stress in MPa.
            allowable_stress (float): Allowable bending stress in MPa. When
                ``None``, the material endurance limit is used.

        Returns:
            float: Bending safety factor.
        """
        allowable = allowable_stress if allowable_stress is not None else self._endurance_mpa()
        return allowable / bending_stress

    def contact_safety_factor(self, contact_stress, allowable_stress):
        """
        Safety factor against surface (pitting) fatigue.

        Args:
            contact_stress (float): Applied AGMA contact stress in MPa.
            allowable_stress (float): Allowable contact stress in MPa.

        Returns:
            float: Contact safety factor.
        """
        return allowable_stress / contact_stress

    def _endurance_mpa(self):
        endurance = self.material_properties["endurance_limit"]
        if endurance is None:
            raise ValueError(
                "Material has no endurance limit; pass allowable_stress explicitly"
            )
        return endurance / 1e6

    def __repr__(self):
        return f"Gear(teeth={self.teeth}, module={self.module}, material={self.material!r})"
