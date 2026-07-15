"""Gear transmissions (trains of two or more gears).

A :class:`Transmission` is built stage by stage; each stage is a mesh of
a driver and a driven element. The driven element of one stage and the
driver of the next are assumed to be mounted on the same shaft (compound
train). Reusing the same gear object in consecutive stages gives a
simple train with idlers.

Planetary sets are composite mechanisms, not two-element meshes, so they
live in :class:`~mecapy.gears.PlanetaryGearSet`; multiply their ratio
with ``overall_ratio`` by hand when combining.
"""

import math

from .bevel import BevelGear
from .cylindrical import CylindricalGear, HerringboneGear
from .rack import Rack
from .worm import Worm, WormWheel


def _check_mesh(driver, driven):
    """
    Validate that two elements can mesh, raising ValueError otherwise.

    Checks the type pairing (spur with spur or rack, helical with
    helical of opposite hand, herringbone with herringbone, bevel with
    bevel, worm with worm wheel), equal module (normal module for
    helical, axial for a worm) and equal pressure angle.

    Args:
        driver: Driving element (Gear subclass or Worm).
        driven: Driven element (Gear subclass or Rack).

    Raises:
        ValueError: With a specific message for each incompatibility.
    """
    # --- type pairing -------------------------------------------------
    if isinstance(driver, Rack):
        raise ValueError("A rack cannot be a driving element")
    if isinstance(driven, Worm):
        raise ValueError("A worm cannot be a driven element")

    if isinstance(driver, Worm):
        if not isinstance(driven, WormWheel):
            raise ValueError(
                f"A worm meshes only with a worm wheel, not "
                f"{type(driven).__name__}"
            )
    elif isinstance(driven, WormWheel):
        raise ValueError("A worm wheel meshes only with a worm")
    elif isinstance(driver, BevelGear) or isinstance(driven, BevelGear):
        if not (isinstance(driver, BevelGear)
                and isinstance(driven, BevelGear)):
            raise ValueError("A bevel gear meshes only with another bevel gear")
    elif isinstance(driver, HerringboneGear) or isinstance(driven,
                                                           HerringboneGear):
        if isinstance(driven, Rack):
            raise ValueError("A herringbone gear cannot mesh with a rack")
        if not (isinstance(driver, HerringboneGear)
                and isinstance(driven, HerringboneGear)):
            raise ValueError(
                "A herringbone gear meshes only with another herringbone gear"
            )
        if not math.isclose(driver.helix_angle, driven.helix_angle):
            raise ValueError(
                "Meshing herringbone gears need equal helix angles"
            )
    elif isinstance(driver, CylindricalGear) and isinstance(driven, Rack):
        pass  # rack and pinion: module/pressure angle checked below
    elif (isinstance(driver, CylindricalGear)
          and isinstance(driven, CylindricalGear)):
        if (driver.helix_angle == 0) != (driven.helix_angle == 0):
            raise ValueError("A spur gear cannot mesh with a helical gear")
        if driver.helix_angle > 0:
            if not math.isclose(driver.helix_angle, driven.helix_angle):
                raise ValueError(
                    "Meshing helical gears need equal helix angles"
                )
            if driver.hand == driven.hand or None in (driver.hand,
                                                      driven.hand):
                raise ValueError(
                    "External helical gears mesh with opposite hands "
                    "(one 'right', one 'left')"
                )
    else:
        raise ValueError(
            f"Cannot mesh {type(driver).__name__} with "
            f"{type(driven).__name__}"
        )

    # --- module and pressure angle ------------------------------------
    if not math.isclose(driver.module, driven.module, rel_tol=1e-6):
        raise ValueError(
            f"Meshing elements need the same module: "
            f"{driver.module} vs {driven.module} mm"
        )
    if not math.isclose(driver.pressure_angle, driven.pressure_angle,
                        rel_tol=1e-6):
        raise ValueError(
            f"Meshing elements need the same pressure angle: "
            f"{driver.pressure_angle} vs {driven.pressure_angle} degrees"
        )


def _stage_ratio(driver, driven):
    """float: Speed ratio of one stage (driver speed / driven speed)."""
    if isinstance(driver, Worm):
        return driven.teeth / driver.starts
    if isinstance(driven, Rack):
        return None  # rotary-to-linear: no dimensionless speed ratio
    return driven.teeth / driver.teeth


class Transmission:
    """
    Multi-stage gear transmission (compound train).

    Build with chained :meth:`add_stage` calls; the driven element of a
    stage shares its shaft with the driver of the next stage.

    Attributes:
        name (str): Optional identifier.
        stages (list): List of (driver, driven) tuples.
    """

    def __init__(self, name=None):
        """
        Initialize an empty transmission.

        Args:
            name (str): Optional identifier.
        """
        self.name = name
        self.stages = []

    def add_stage(self, driver, driven):
        """
        Append a mesh stage to the train.

        Args:
            driver: Driving element (gear or worm). Must not follow a
                rack stage.
            driven: Driven element (gear, worm wheel or rack). A rack
                ends the train.

        Returns:
            Transmission: self, for chaining.

        Raises:
            ValueError: If the pair is incompatible or a rack is not
                the final element.
        """
        if self.stages and isinstance(self.stages[-1][1], Rack):
            raise ValueError("A rack must be the final element of the train")
        _check_mesh(driver, driven)
        self.stages.append((driver, driven))
        return self

    # ------------------------------------------------------------------
    # Ratios and kinematics
    # ------------------------------------------------------------------

    @property
    def ends_in_rack(self):
        """bool: True if the last driven element is a rack."""
        return bool(self.stages) and isinstance(self.stages[-1][1], Rack)

    @property
    def overall_ratio(self):
        """float: Overall speed reduction ratio (input speed / output
        speed), product of the stage ratios. Excludes a final rack
        stage (rotary to linear).

        Raises:
            ValueError: If no stage has been added.
        """
        self._require_stages()
        ratio = 1.0
        for driver, driven in self.stages:
            r = _stage_ratio(driver, driven)
            if r is not None:
                ratio *= r
        return ratio

    @property
    def train_value(self):
        """float: Signed train value (output speed / input speed) for
        parallel-axis trains. Each external cylindrical mesh reverses
        the direction (negative sign). None if the train contains
        bevel or worm stages (axis changes) or ends in a rack.
        """
        self._require_stages()
        value = 1.0
        for driver, driven in self.stages:
            if isinstance(driver, (Worm, BevelGear)) or isinstance(
                    driven, (Rack, BevelGear, WormWheel)):
                return None
            value *= -driver.teeth / driven.teeth
        return value

    def output_speed(self, input_speed_rpm):
        """
        Output speed for a given input speed.

        Args:
            input_speed_rpm (float): Speed of the first driver in rpm.

        Returns:
            float: Output speed in rpm — or, if the train ends in a
            rack, the rack linear velocity in m/s.
        """
        self._require_stages()
        speed = input_speed_rpm / self.overall_ratio
        if self.ends_in_rack:
            pinion, rack = self.stages[-1]
            return rack.linear_velocity(pinion, speed)
        return speed

    def output_torque(self, input_torque, efficiency=1.0):
        """
        Output torque for a given input torque (rotary output only).

        Args:
            input_torque (float): Torque on the first driver (any unit).
            efficiency (float): Overall mechanical efficiency, 0 < eta
                <= 1 (default 1.0; multiply per-stage efficiencies —
                e.g. :meth:`Worm.efficiency` — yourself).

        Returns:
            float: Output torque in the same unit as the input.

        Raises:
            ValueError: If the train ends in a rack (use force balance
                instead) or the efficiency is out of range.
        """
        if self.ends_in_rack:
            raise ValueError(
                "The train ends in a rack; torque becomes a linear force"
            )
        if not 0 < efficiency <= 1:
            raise ValueError("Efficiency must be in (0, 1]")
        return input_torque * self.overall_ratio * efficiency

    def speeds(self, input_speed_rpm):
        """
        Rotational speed of every element, stage by stage.

        Args:
            input_speed_rpm (float): Speed of the first driver in rpm.

        Returns:
            list: One dict per stage with keys "driver_rpm" and
            "driven_rpm" ("driven_m_s" for a final rack).
        """
        self._require_stages()
        result = []
        speed = float(input_speed_rpm)
        for driver, driven in self.stages:
            entry = {"driver_rpm": speed}
            if isinstance(driven, Rack):
                entry["driven_m_s"] = driven.linear_velocity(driver, speed)
            else:
                speed = speed / _stage_ratio(driver, driven)
                entry["driven_rpm"] = speed
            result.append(entry)
        return result

    def torques(self, input_torque, efficiency=1.0):
        """
        Torque on every shaft, stage by stage (ideal or with a global
        efficiency applied per stage).

        Args:
            input_torque (float): Torque on the first driver.
            efficiency (float): Per-stage efficiency (default 1.0).

        Returns:
            list: One dict per stage with keys "driver_torque" and
            "driven_torque" ("driven_force_n" for a final rack, torque
            in N*mm assumed).
        """
        self._require_stages()
        if not 0 < efficiency <= 1:
            raise ValueError("Efficiency must be in (0, 1]")
        result = []
        torque = float(input_torque)
        for driver, driven in self.stages:
            entry = {"driver_torque": torque}
            if isinstance(driven, Rack):
                # Torque (N*mm) to force (N) at the pinion pitch radius.
                entry["driven_force_n"] = (2 * torque * efficiency
                                           / driver.pitch_diameter)
            else:
                torque = torque * _stage_ratio(driver, driven) * efficiency
                entry["driven_torque"] = torque
            result.append(entry)
        return result

    def center_distance(self, stage_index):
        """
        Center distance of one stage (parallel-axis or worm stages).

        Args:
            stage_index (int): Zero-based stage index.

        Returns:
            float: Center distance in mm.

        Raises:
            ValueError: For a rack or bevel stage (no center distance).
        """
        driver, driven = self.stages[stage_index]
        if isinstance(driven, Rack):
            raise ValueError("A rack stage has no center distance")
        if isinstance(driver, BevelGear):
            raise ValueError("A bevel stage has intersecting axes")
        return (getattr(driver, "pitch_diameter")
                + driven.pitch_diameter) / 2

    def pitch_line_velocity(self, stage_index, input_speed_rpm):
        """
        Pitch-line velocity at one stage for a given train input speed.

        Args:
            stage_index (int): Zero-based stage index.
            input_speed_rpm (float): Speed of the first driver in rpm.

        Returns:
            float: Pitch-line velocity in m/s at that stage's mesh.
        """
        self._require_stages()
        speed = float(input_speed_rpm)
        for i, (driver, driven) in enumerate(self.stages):
            if i == stage_index:
                d = driver.pitch_diameter
                return math.pi * d * speed / 60000.0
            r = _stage_ratio(driver, driven)
            if r is None:
                raise ValueError("No rotary stage after a rack")
            speed = speed / r
        raise ValueError(f"Stage index {stage_index} out of range")

    def _require_stages(self):
        if not self.stages:
            raise ValueError("The transmission has no stages")

    def __repr__(self):
        if not self.stages:
            return f"Transmission(name={self.name!r}, stages=0)"
        desc = " -> ".join(
            f"{type(d).__name__}/{type(n).__name__}" for d, n in self.stages
        )
        try:
            ratio = f", ratio={self.overall_ratio:.3f}"
        except ValueError:
            ratio = ""
        return f"Transmission({desc}{ratio})"
