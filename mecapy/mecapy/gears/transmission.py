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
from .gear import Gear
from .rack import Rack
from .worm import Worm, WormWheel


def _check_mesh(driver, driven):
    """
    Validate that two elements can mesh, raising ValueError otherwise.
    Checks the type pairing (spur with spur or rack, helical with
    helical of opposite hand, herringbone with herringbone, bevel with
    bevel, worm with worm wheel), equal module (normal module for
    helical, axial for a worm) and equal pressure angle. Profile shift
    does not affect mesh compatibility (it only moves the working
    center distance).

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


def _stage_pinion(driver, driven):
    """
    Order one stage as (pinion, gear) for an AGMA rating.

    :class:`~mecapy.gears.agma.AGMARating` requires the pinion to be the
    smaller member, so a speed-increasing stage (driven smaller than the
    driver) is swapped. A rack is always the mating member.

    Args:
        driver: Driving element of the stage.
        driven: Driven element of the stage.

    Returns:
        tuple or None: ``(pinion, gear)`` for an AGMA-rateable stage,
        else ``None`` (bevel and worm stages are outside AGMA 2101-D04).
    """
    if not isinstance(driver, CylindricalGear):
        return None
    if isinstance(driven, Rack):
        return (driver, driven)
    if not isinstance(driven, CylindricalGear):
        return None
    if driven.teeth < driver.teeth:
        return (driven, driver)
    return (driver, driven)


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
        self._stage_efficiencies = []

    def add_stage(self, driver, driven, efficiency=1.0):
        """
        Append a mesh stage to the train.

        Args:
            driver: Driving element (gear or worm). Must not follow a
                rack stage.
            driven: Driven element (gear, worm wheel or rack). A rack
                ends the train.
            efficiency (float): Mechanical efficiency of this stage,
                0 < eta <= 1 (default 1.0 = lossless). Power passed to
                the next shaft is multiplied by this factor.

        Returns:
            Transmission: self, for chaining.

        Raises:
            ValueError: If the pair is incompatible, a rack is not the
                final element, or the efficiency is out of range.
        """
        if self.stages and isinstance(self.stages[-1][1], Rack):
            raise ValueError("A rack must be the final element of the train")
        if not 0 < efficiency <= 1:
            raise ValueError("Efficiency must be in (0, 1]")
        _check_mesh(driver, driven)
        self.stages.append((driver, driven))
        self._stage_efficiencies.append(efficiency)
        self._try_propagate()
        return self

    def propagate(self):
        """
        Push the first gear's operating point through the whole train.

        Starting from the first driver's ``power_kw`` and ``speed_rpm``,
        assign each downstream element its rotational speed (input speed
        divided by the cumulative stage ratio) and its transmitted power
        (multiplied by each stage efficiency). Elements sharing a shaft
        take the same speed and power. Non-gear elements (a rack or a
        worm) are skipped — they carry no rotational operating point.

        Call this again after editing a gear in place (e.g.
        :meth:`Gear.change_teeth`) to recompute the train — the values
        are stored on the gears, so they do not update on their own.

        Returns:
            Transmission: self, for chaining.

        Raises:
            ValueError: If there are no stages, or the first driver has
                no ``power_kw``/``speed_rpm`` set.
        """
        self._require_stages()
        first = self.stages[0][0]
        if getattr(first, "power_kw", None) is None or \
                getattr(first, "speed_rpm", None) is None:
            raise ValueError(
                "The first driver must have power_kw and speed_rpm set "
                "before the transmission can propagate them"
            )
        speed, power = first.speed_rpm, first.power_kw
        for (driver, driven), eff in zip(self.stages,
                                         self._stage_efficiencies):
            if isinstance(driver, Gear):
                driver.speed_rpm = speed
                driver.power_kw = power
            r = _stage_ratio(driver, driven)
            if r is not None:
                speed = speed / r
            power = power * eff
            if isinstance(driven, Gear):
                driven.speed_rpm = speed
                driven.power_kw = power
        return self

    def _try_propagate(self):
        """Propagate only if the first driver already has power and speed."""
        first = self.stages[0][0]
        if getattr(first, "power_kw", None) is not None and \
                getattr(first, "speed_rpm", None) is not None:
            self.propagate()

    @property
    def output_power(self):
        """float: Power at the output shaft in kW (first gear's power
        times every stage efficiency).

        Raises:
            ValueError: If there are no stages or the first driver has
                no ``power_kw`` set.
        """
        self._require_stages()
        first = self.stages[0][0]
        if getattr(first, "power_kw", None) is None:
            raise ValueError("The first driver has no power_kw set")
        power = first.power_kw
        for eff in self._stage_efficiencies:
            power *= eff
        return power

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

    # ------------------------------------------------------------------
    # AGMA rating of the whole train
    # ------------------------------------------------------------------

    @property
    def agma_unrated_stages(self):
        """list: ``(stage_index, reason)`` for every stage AGMA 2101-D04
        does not cover (bevel and worm stages). Empty for an all-
        cylindrical train.
        """
        unrated = []
        for i, (driver, driven) in enumerate(self.stages):
            if _stage_pinion(driver, driven) is None:
                unrated.append((
                    i,
                    f"{type(driver).__name__}/{type(driven).__name__} stage - "
                    f"AGMA 2101-D04 covers cylindrical gears only"
                ))
        return unrated

    #: Idler bending factor applied to a gear that meshes in more than
    #: one stage: both flanks of its teeth are loaded each revolution
    #: (fully reversed bending) instead of only one (repeated), which
    #: needs roughly 1/0.70 of the one-directional bending strength.
    _IDLER_KI = 1.42

    def _idler_gears(self):
        """set: ``id()`` of every element that appears in more than one
        stage of the train (reused as the driven of one stage and the
        driver of the next — an idler). Used to set ``Ki_pinion``/
        ``Ki_gear`` automatically in :meth:`rate_agma`.
        """
        counts = {}
        for driver, driven in self.stages:
            for element in (driver, driven):
                counts[id(element)] = counts.get(id(element), 0) + 1
        return {key for key, n in counts.items() if n > 1}

    #: Rating inputs the train supplies itself; they cannot be overridden
    #: per stage because they come from :meth:`propagate`.
    _TRAIN_SUPPLIED = ("power_kw", "pinion_speed_rpm", "tangential_force")

    def _merged_stage_kwargs(self, stage_kwargs, kwargs):
        """Validate ``stage_kwargs`` and merge it over the global kwargs."""
        if stage_kwargs is None:
            return [dict(kwargs) for _ in self.stages]
        if len(stage_kwargs) != len(self.stages):
            raise ValueError(
                f"stage_kwargs must have one entry per stage "
                f"({len(self.stages)}), got {len(stage_kwargs)}"
            )
        for i, override in enumerate(stage_kwargs):
            clash = set(override or ()) & set(self._TRAIN_SUPPLIED)
            if clash:
                raise ValueError(
                    f"stage_kwargs[{i}] cannot set {sorted(clash)}: the "
                    f"transmission supplies the operating point of every "
                    f"stage from propagate()"
                )
        return [{**kwargs, **(s or {})} for s in stage_kwargs]

    def rate_agma(self, stage_kwargs=None, Ko=1.0, Qv=7, Ks=None, ZR=1.0,
                  KB=1.0, life_cycles=1e7, reliability=0.99, grade=1,
                  hardness_HB=None, gear_hardness_HB=None, St=None, Sc=None,
                  YJ_pinion=None, YJ_gear=None, condition="commercial",
                  crowned=False, temperature_factor=1.0,
                  temperature_celsius=None):
        """
        AGMA bending and pitting rating of every stage of the train.

        Calls :meth:`propagate` first, so each mesh is rated at its own
        propagated speed and power (including the stage efficiencies).
        The smaller member of each stage is taken as the pinion. Stages
        AGMA 2101-D04 does not cover (bevel, worm) are skipped — see
        :attr:`agma_unrated_stages`.

        Every rating input of :class:`~mecapy.gears.agma.AGMARating` is
        accepted here except the operating point (``power_kw``,
        ``pinion_speed_rpm``, ``tangential_force``), which the train
        supplies per stage. Values given here apply to all stages;
        ``stage_kwargs`` overrides them stage by stage.

        ``Ki_pinion``/``Ki_gear`` (idler bending factor) are also set
        automatically per stage: 1.42 for a member that appears in more
        than one stage of the train (an idler — the same gear object
        reused as driven of one stage and driver of the next), else 1.0.
        Only the idler member gets the factor, not its mate. Override
        with ``stage_kwargs`` if a specific stage needs a different value.

        Note: the AGMA J-factor tables assume standard (profile shift
        x = 0) tooth proportions; ratings for shifted gears are
        approximate.

        Args:
            stage_kwargs (list): Optional per-stage keyword overrides,
                one dict (or None) per stage, indexed by stage index —
                including unrated stages, so the indices stay aligned.
                Entries override the arguments below.
            Ko (float): Overload factor (default 1.0). Typical values:
                uniform/uniform 1.0, light shock 1.25, moderate shock
                1.5, heavy shock 1.75+.
            Qv (int): AGMA quality number for Kv, 3 to 11 (default 7).
            Ks (float): Size factor. Default ``None`` computes it per
                stage from that stage's pinion module via
                :func:`~mecapy.gears.agma.size_factor` — so a coarse
                output stage is penalised more than a fine input stage.
                Give a number to override the table everywhere.
            ZR (float): Surface-condition factor (default 1.0).
            KB (float): Rim-thickness factor (default 1.0, solid gear);
                see :func:`~mecapy.gears.agma.rim_thickness_factor`.
            life_cycles (float): Load cycles for YN/ZN (default 1e7).
                Note that a downstream stage turns fewer cycles for the
                same running time — set it per stage when that matters.
            reliability (float): Survival probability for YZ (default
                0.99).
            grade (int): AGMA steel grade for the allowable-stress fits
                (default 1). Ignored for a gear whose material is a
                :class:`~mecapy.gears.GearMaterial` supplying its own.
            hardness_HB (float): Brinell hardness for the through-
                hardened allowable-stress fits. Give this OR explicit
                ``St``/``Sc``; defaults to each pinion's own
                :class:`~mecapy.gears.GearMaterial` hardness.
            gear_hardness_HB (float): Gear hardness if different from
                the pinion (enables the ZW factor).
            St (float): Explicit allowable bending stress number in MPa.
            Sc (float): Explicit allowable contact stress number in MPa.
            YJ_pinion (float): Override for the pinion bending geometry
                factor J. Tooth-count specific, so it usually belongs in
                ``stage_kwargs`` rather than here.
            YJ_gear (float): Override for the gear J, same caveat.
            condition (str): Mounting condition for KH — "open",
                "commercial" (default), "precision" or "extra_precision".
            crowned (bool): Crowned teeth for KH (default False).
            temperature_factor (float): Explicit Y_theta (default 1.0).
                Give this OR ``temperature_celsius``, not both.
            temperature_celsius (float): Operating (oil) temperature in
                degrees Celsius, converted to Y_theta.

        Returns:
            list: One :class:`~mecapy.gears.agma.AGMARating` per rateable
            stage, each carrying a ``stage_index`` attribute. Every
            stress and factor on them stays live (recomputed on access).

        Raises:
            ValueError: If there are no stages, the first driver has no
                power/speed, ``stage_kwargs`` has the wrong length or
                tries to set the operating point, or a stage cannot be
                rated (message prefixed with its index).
        """
        from .agma import AGMARating

        kwargs = dict(
            Ko=Ko, Qv=Qv, Ks=Ks, ZR=ZR, KB=KB, life_cycles=life_cycles,
            reliability=reliability, grade=grade, hardness_HB=hardness_HB,
            gear_hardness_HB=gear_hardness_HB, St=St, Sc=Sc,
            YJ_pinion=YJ_pinion, YJ_gear=YJ_gear, condition=condition,
            crowned=crowned, temperature_factor=temperature_factor,
            temperature_celsius=temperature_celsius,
        )
        self.propagate()
        merged = self._merged_stage_kwargs(stage_kwargs, kwargs)
        idlers = self._idler_gears()
        ratings = []
        for i, (driver, driven) in enumerate(self.stages):
            pair = _stage_pinion(driver, driven)
            if pair is None:
                continue
            pinion, gear = pair
            stage_args = dict(merged[i])
            stage_args.setdefault(
                "Ki_pinion", self._IDLER_KI if id(pinion) in idlers else 1.0)
            if not isinstance(gear, Rack):
                stage_args.setdefault(
                    "Ki_gear", self._IDLER_KI if id(gear) in idlers else 1.0)
            try:
                rating = AGMARating(pinion, gear,
                                    power_kw=pinion.power_kw,
                                    pinion_speed_rpm=pinion.speed_rpm,
                                    **stage_args)
            except ValueError as exc:
                raise ValueError(f"Stage {i}: {exc}") from exc
            rating.stage_index = i
            ratings.append(rating)
        return ratings

    def agma_governing(self, stage_kwargs=None, Ko=1.0, Qv=7, Ks=None,
                       ZR=1.0, KB=1.0, life_cycles=1e7, reliability=0.99,
                       grade=1, hardness_HB=None, gear_hardness_HB=None,
                       St=None, Sc=None, YJ_pinion=None, YJ_gear=None,
                       condition="commercial", crowned=False,
                       temperature_factor=1.0, temperature_celsius=None):
        """
        Worst bending and pitting safety factor over the whole train.

        Bending and pitting may be governed by different stages, so each
        reports its own stage index.

        Args:
            stage_kwargs (list): Per-stage overrides, see :meth:`rate_agma`.
            Ko, Qv, Ks, ZR, KB, life_cycles, reliability, grade,
                hardness_HB, gear_hardness_HB, St, Sc, YJ_pinion, YJ_gear,
                condition, crowned, temperature_factor,
                temperature_celsius: Rating arguments, forwarded to
                :meth:`rate_agma` (see its Args for units and defaults).

        Returns:
            dict: Keys "SF" (minimum bending safety factor over both
            members of every stage), "SF_stage", "SH" (minimum pitting
            safety factor) and "SH_stage".

        Raises:
            ValueError: If no stage of the train can be rated.
        """
        ratings = self.rate_agma(
            stage_kwargs, Ko=Ko, Qv=Qv, Ks=Ks, ZR=ZR, KB=KB,
            life_cycles=life_cycles, reliability=reliability, grade=grade,
            hardness_HB=hardness_HB, gear_hardness_HB=gear_hardness_HB,
            St=St, Sc=Sc, YJ_pinion=YJ_pinion, YJ_gear=YJ_gear,
            condition=condition, crowned=crowned,
            temperature_factor=temperature_factor,
            temperature_celsius=temperature_celsius,
        )
        if not ratings:
            raise ValueError("No stage of this transmission is AGMA-rateable")
        best_sf = best_sh = None
        for r in ratings:
            for sf in (r.SF_pinion, r.SF_gear):
                if sf is not None and (best_sf is None or sf < best_sf[0]):
                    best_sf = (sf, r.stage_index)
            if best_sh is None or r.SH < best_sh[0]:
                best_sh = (r.SH, r.stage_index)
        return {"SF": best_sf[0], "SF_stage": best_sf[1],
                "SH": best_sh[0], "SH_stage": best_sh[1]}

    def agma_summary(self, stage_kwargs=None, Ko=1.0, Qv=7, Ks=None,
                     ZR=1.0, KB=1.0, life_cycles=1e7, reliability=0.99,
                     grade=1, hardness_HB=None, gear_hardness_HB=None,
                     St=None, Sc=None, YJ_pinion=None, YJ_gear=None,
                     condition="commercial", crowned=False,
                     temperature_factor=1.0, temperature_celsius=None):
        """
        Human-readable AGMA rating of the whole train.

        Args:
            stage_kwargs (list): Per-stage overrides, see :meth:`rate_agma`.
            Ko, Qv, Ks, ZR, KB, life_cycles, reliability, grade,
                hardness_HB, gear_hardness_HB, St, Sc, YJ_pinion, YJ_gear,
                condition, crowned, temperature_factor,
                temperature_celsius: Rating arguments, forwarded to
                :meth:`rate_agma` (see its Args for units and defaults).

        Returns:
            str: Multi-line report: one
            :meth:`~mecapy.gears.agma.AGMARating.summary` per rateable
            stage, the unrated stages, and the governing safety factors.
        """
        kwargs = dict(
            Ko=Ko, Qv=Qv, Ks=Ks, ZR=ZR, KB=KB, life_cycles=life_cycles,
            reliability=reliability, grade=grade, hardness_HB=hardness_HB,
            gear_hardness_HB=gear_hardness_HB, St=St, Sc=Sc,
            YJ_pinion=YJ_pinion, YJ_gear=YJ_gear, condition=condition,
            crowned=crowned, temperature_factor=temperature_factor,
            temperature_celsius=temperature_celsius,
        )
        ratings = self.rate_agma(stage_kwargs, **kwargs)
        lines = [
            f"Transmission AGMA Rating - {self.name or 'unnamed'}",
            "=" * 40,
            f"Stages: {len(self.stages)}, rated: {len(ratings)}",
            "",
        ]
        for r in ratings:
            lines.append(f"--- Stage {r.stage_index} ---")
            lines.append(r.summary())
            lines.append("")
        for index, reason in self.agma_unrated_stages:
            lines.append(f"Stage {index} not rated: {reason}")
        if ratings and all(r.has_allowables for r in ratings):
            g = self.agma_governing(stage_kwargs, **kwargs)
            lines += [
                "=" * 40,
                f"Governing bending: SF={g['SF']:.2f} "
                f"(stage {g['SF_stage']})",
                f"Governing pitting: SH={g['SH']:.2f} "
                f"(stage {g['SH_stage']})",
            ]
        elif ratings:
            lines += [
                "=" * 40,
                f"Material selection (SF = SH = 1): every stage needs "
                f"St >= {max(r.required_St() for r in ratings):.1f} MPa "
                f"and Sc >= {max(r.required_Sc() for r in ratings):.1f} MPa",
            ]
        return "\n".join(lines)

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
