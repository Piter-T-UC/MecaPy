# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

The Python project lives in the **`MecaPy/`** subdirectory: package code in
`MecaPy/mecapy/`, tests in `MecaPy/tests/`, plus `docs/`, `examples/` and the
graphify graph (`MecaPy/graphify-out/`). Run all development commands from
`MecaPy/`. Paths elsewhere in this document are relative to `MecaPy/`.

## Quick Start: Understanding the Codebase

The fastest way to understand how things connect is via **graphify** — a knowledge graph built from the codebase:

```bash
# Query existing graph (no rebuild, instant answers)
/graphify query "how does AGMA rating calculate stress?"
/graphify query "what's the difference between spur and helical gears?"
/graphify path "Transmission" "SpurGear"     # shortest path between concepts
/graphify explain "PlanetaryGearSet"         # plain-language summary

# Update graph after major code changes
/graphify . --update

# Full rebuild if structure changed dramatically
/graphify .
```

Open `MecaPy/graphify-out/graph.html` in your browser for an interactive visualization.

## Commands

```bash
# All commands run from the project directory
cd MecaPy

# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run the full test suite
pytest

# Run a single test file / class / test
pytest tests/test_gears.py
pytest tests/test_gears.py::TestSpurGear
pytest tests/test_gears.py::TestSpurGear::test_geometry

# Coverage
pytest --cov=mecapy --cov-report=html

# Lint / format / type-check
black .
flake8 .
mypy mecapy/

# Build docs (Sphinx)
cd docs && make html   # open _build/html/index.html
```

Note: `mecapy` is not installed in this environment by default — scripts run directly
(e.g. `python examples/gear_design.py`) need `PYTHONPATH=.` (with `MecaPy/` as the
working directory) unless `pip install -e .` has been run first.

flake8 config: max line length 100, ignores E203/W503 (`setup.cfg`). mypy: Python 3.8 target,
`disallow_untyped_defs = False` — the codebase does not use type annotations, relying on
Google-style docstrings instead.

## Architecture

Every mechanical component inherits from a single base class, `MechaElement`
(`mecapy/base.py`), which provides shared material access and the fundamental
stress/safety-factor calculations:

```
MechaElement                              # material_properties, calculate_stress(F, A), safety_factor(stress)
├── Beam (beams/)                         # SymPy-backed: reactions, shear, moment, deflection
├── Shaft (shafts/)                       # + torsional_stress()
│   └── PowerScrew (shafts/power_screw.py)  # lead screw: thread stresses, buckling (composes Column), efficiency
├── Column (columns/)                     # Euler/Johnson strut buckling, secant formula (Shigley Ch. 4)
├── Key / Pin / Rivet (joints/)           # torque/shear connections: shear+bearing checks (Shigley Ch. 7-8)
├── Gear (gears/)                         # base gear geometry; see gear type hierarchy below
├── Wheel (wheels/)                       # pulleys, sprockets; + moment_of_inertia, kinetic_energy
│   └── Flywheel (wheels/flywheel.py)     # energy fluctuation sizing, rotating-disc stresses (SI)
├── Bearing (bearings/)                   # rolling contact stress & life
├── Bolt (bolts/)                         # ISO metric: tension, preload, stiffness
│   └── BoltedUnion (bolts/)              # joint: load distribution, preload chain, member checks
├── AxialFrictionInterface (clutches/)    # annular friction math: uniform pressure & uniform wear
│   ├── DiscClutch                        # flat disc (= full-annulus disc brake, alias DiscBrake)
│   └── ConeClutch                        # wedging cone, 1/sin(alpha) torque amplification
├── CentrifugalClutch (clutches/)         # spring-retained shoes, engagement speed, T(omega)
├── InternalShoeBrake / ExternalShoeBrake (brakes/shoe.py)  # pivoted long shoe (Shigley 16-2..16-8)
├── BandBrake (brakes/band.py)            # e^(mu*phi) tension ratio, band stress
├── CaliperDiscBrake (brakes/disc.py)     # annular-sector pads, both wear theories
├── FlangeCoupling (couplings/)           # rigid: bolt/key/flange checks, torque_capacity()
├── FlexibleCoupling (couplings/)         # catalog-style: torque/speed/misalignment ratings
├── FlatBelt (belts/flat.py)              # capstan e^(mu*phi) drive, tension/power round-trip
│   └── VBelt (belts/vbelt.py)            # groove wedging, 1/sin(groove_angle/2) friction amplification
├── RollerChain (chains/roller.py)        # ANSI chain kinematics, tensile SF, H1/H2 rated power
└── Weld (welds/)                         # weld bead stresses & material
    └── WeldedUnion (welds/)              # weld group: combined loading, sizing, plots
```

Because every element shares `MechaElement`, any of them can compute an axial stress
and a safety factor against yielding via the same two inherited methods — there is no
per-element stress/safety duplication.

### How Stress & Safety Work Everywhere

Every element inherits two methods from `MechaElement`:
- **`calculate_stress(force, area)`** — returns MPa for a given force/area pair
- **`safety_factor(stress)`** — returns ratio of yield strength to actual stress

This means you can check if a Bolt is safe the same way you check a Gear tooth:

```python
bolt = Bolt("M10", 50, material="steel")
stress = bolt.calculate_stress(8000, bolt.stress_area)  # 8 kN on M10 bolt
sf = bolt.safety_factor(stress)
assert sf > 1.5  # meets safety requirement
```

Materials are looked up by name (`"steel"`, `"aluminum"`, `"copper"`, `"cast_iron"`,
`"bronze"`) from a database in `mecapy/utils/constants.py` (`MATERIALS` dict), accessed
through `mecapy/materials.py` (`get_material_properties`, `get_available_materials`,
`add_custom_material`). All material properties are strict SI (Pa, kg/m^3).

## Subsystem Overview: What Does Each Module Do?

### Shafts (`mecapy/shafts/`)
**Purpose:** rotational members carrying bending and torsional loads.

- **`Shaft`** — circular shaft with bending/torsional stress, deflection, critical speed
  - Computes combined bending + torsion using von Mises equivalent stress
  - Used in gearbox design to size intermediate shafts
- **`PowerScrew`** — lead screw (translating screw for power transmission)
  - Models thread contact stresses, buckling under axial load
  - Calculates raising/lowering torque with optional collar friction
  - Thread forms: square, Acme, trapezoidal, buttress, or custom
  - Efficiency & self-locking analysis

### Gears (`mecapy/gears/`) — Largest subsystem
**Purpose:** power transmission through tooth mesh.

- **`Gear`** — base class, standard full-depth involute geometry
  - Common to all types: pitch/base/outside/root diameters (+ `*_radius` accessors),
    addendum/dedendum, clearance, working/whole depth, tooth thickness, base pitch
  - `describe()` returns a multi-line report of every parameter with symbol and unit
    (e.g. `addendum (ha) = 2.500 mm`); subclasses extend it (helix block, undercut check)
  - `involute()` / `inverse_involute()` helpers live in `gear.py` and are exported
  - Accepts `module` (mm, SI) OR `diametral_pitch` (teeth/inch) — exactly one required
  - All downstream code works in metric internally
- **Profile shift** (cylindrical gears only): pass `profile_shift=x` (-1 < x < 1) to any
  cylindrical gear — addendum m(1+x), dedendum m(1.25-x), tooth thickness follow.
  `min_profile_shift` / `is_undercut` check undercut; pair methods
  `working_pressure_angle_with()`, `working_center_distance_with()` and
  `has_interference_with()` handle shifted meshes (`center_distance_with()` stays the
  reference distance). AGMA tables assume x = 0 (ratings for shifted gears are approximate).
- **`CylindricalGear`** family:
  - **`SpurGear`** — parallel axes, no helix angle
  - **`HelicalGear`** — parallel axes with helix angle (requires hand: "right"/"left")
  - **`HerringboneGear`** — back-to-back helical, thrust cancels automatically
- **`BevelGear`** — intersecting axes (cone geometry)
  - Pair-wise methods (`cone_angle_with()`, `contact_ratio_with()`) require the mating gear
- **`Worm` + `WormWheel`** — non-parallel, non-intersecting axes
  - Worm has `starts` and `lead` instead of teeth/module
  - WormWheel is a modified gear that meshes only with Worm
- **`Rack`** — infinite pitch radius (linear motion)
  - Terminal member of a transmission (cannot drive, only be driven)
- **`PlanetaryGearSet`** — composite of sun/planet/ring
  - Solves via Willis train-value equation
  - Not a `Transmission` stage (three members, not two)
- **`Transmission`** — chains 2+ meshes together
  - Single source of truth for mesh compatibility (`_check_mesh()`)
  - Kinematic model: speed ratios, output torque/power
  - Reused by AGMA rating and other pair methods
  - Train-level AGMA: `rate_agma(stage_kwargs=None, **kwargs)` returns one
    `AGMARating` per cylindrical stage (each tagged with `stage_index`, rated at
    its own propagated speed/power); `agma_governing()` gives the worst SF/SH and
    which stage they come from, `agma_summary()` prints the whole train.
    Bevel/worm stages are skipped and listed by `agma_unrated_stages`.
- **AGMA Rating** (`agma.py` + `agma_data.py`) — fatigue & contact stress
  - Implements AGMA 2101-D04 (metric) for cylindrical gears
  - Bending stress (Lewis + geometry factor) and pitting (Hertzian contact)
  - Digitized tables: Lewis Y, J geometry factors, material allowables, reliability factors
  - Module-level factor functions in `agma.py`: `dynamic_factor`, `load_distribution_factor`,
    `geometry_factor_I` (Norton's surface geometry factor), `elastic_coefficient`,
    `bending_life_factor`, `contact_life_factor`, `temperature_factor`,
    `hardness_ratio_factor`, `rim_thickness_factor`
  - Accepts explicit overrides to bypass table lookup
- **Force reports** — every gear type (`Gear`, `BevelGear`, `Rack`, `Worm`, `WormWheel`)
  has a `force_report(power_kw, speed_rpm, ...)` method returning the tangential /
  radial / axial force breakdown at the mesh

### Bolts (`mecapy/bolts/`)
**Purpose:** threaded fasteners & bolted joints.

- **`Bolt`** — ISO metric bolt
  - Thread geometry from ISO 898-1 coarse table (diameter, pitch, stress area)
  - Strength data from ISO 898-1 property classes (8.8, 10.9, etc.)
  - Preload capacity, tensile stress calculation
  - **Two stiffnesses, deliberately.** `stiffness` is the free-length spring
    `As*E/L` behind `elongation()`; `segmented_stiffness(grip)` is Shigley
    Eq. 8-17 `kb = Ad*At*E/(Ad*lt + At*ld)`, splitting the grip into unthreaded
    shank and thread via the Table 8-7 rule (`threaded_length` in `thread_data.py`,
    also `Bolt.threaded_length` / `shank_length`). Joint analysis always uses the
    segmented one
- **`BoltedUnion`** — multi-bolt joint
  - Models load distribution across bolt group
  - Eccentric load handling (moment splitting)
  - Bending Mx/My resolves about the **extreme bolt** on the compression side by
    default (`bending_reference="extreme"`): the plate cannot pull, so lever arms
    are measured from that pivot bolt and every bolt takes tension (`>= 0`, zero
    at the pivot), the compression being reacted by plate bearing there. Torsion
    Mz always uses the centroid. `bending_reference="centroid"` restores the
    classic centroidal-axis split (tension half / compression half, Fz-balanced).
    `bending_pivots` names the pivot bolt per axis; `bolt_forces()` breaks the
    axial load down into `axial_direct` / `axial_bending_x` / `axial_bending_y`
  - Joint separation & slip safety factors
  - **Preload chain** (needs `plates`): `bolt_stiffness` (segmented, over
    `effective_grip`), `member_stiffness` (30° frusta), `joint_constant`
    C = kb/(kb+km), `effective_preload`. `bolt_tensions()` returns per bolt the
    external load P (which already includes the Fz share *and* the Mx/My bending
    share), the resultant bolt load `Fb = Fi + C*P` and the member load
    `Fm = (1-C)*P - Fi`
  - **Tapped / blind holes**: `tapped=True` makes the last member tapped instead
    of nutted. `grip` stays the physical stack; `effective_grip` is Shigley's
    `l' = h + t2/2` (or `h + d/2` when `t2 >= d`) and is what the whole stiffness
    chain uses. Bolt length is validated both ways — it must reach in without
    bottoming out
  - **Member-side checks**: `bearing_stresses()` / `bearing_safety_factors()`
    (`V/(d*t)` on each plate, bolt nominal `d` bears — not the hole),
    `clamp_states()` (clamp `-Fm`, `separated` flag, washer-face pressure `Fb/Aw`
    over the `dw = 1.5*d` annulus), and `minimum_edge_distances(safety_factor)`
    which reports the tear-out edge distance you must *provide* rather than
    demanding one as input
  - `describe()` / `joint_report()` summarize the whole chain (both degrade
    gracefully without plates); `plot_distribution()` visualizes the per-bolt
    shear breakdown and `plot_tension()` the preload/bolt-load/clamp bars

### Beams (`mecapy/beams/`)
**Purpose:** static & dynamic analysis of bending members.

- **`Beam`** — SymPy-backed symbolic solver
  - Builder pattern: `add_support()`, `add_load()`, `add_moment()`, etc. return `self`
  - Lazy solve: reactions computed only when needed
  - Results: shear force, bending moment, deflection, stress
- **Standalone helpers** (`calculations.py`):
  - Cantilever deflection, bending stress (M*c/I), shear stress (V/A)
  - Useful when you don't need a full solver

### Wheels (`mecapy/wheels/`)
**Purpose:** general rotating members (pulleys, sprockets, flywheels).

- **`Wheel`** — base rotating member with inertia & stress
  - Rim stresses under centrifugal loading
  - Moment of inertia calculation

### Welds (`mecapy/welds/`)
**Purpose:** weld runs and weld groups under combined loading.

- **`Weld`** — a single weld run with electrode material
  - Electrode strengths from `electrode_data.py` (`ELECTRODES` table, `get_electrode()`)
- **`WeldLine` / `WeldCircle`** (`geometry.py`) — weld path geometries; subclass `WeldPath` to add new shapes
- **`WeldedUnion`** — weld group analysed by the elastic "weld as a line" method (Shigley Ch. 9)
  - Group properties: centroid, unit second/polar moments, throat area
  - `weld_stresses()` / `max_stress()` / `safety_factors()` sample the stress around the paths
  - `required_size(safety_factor=..., apply=True)` sizes the weld leg for a target safety factor
  - `plot_distribution()` renders the stress distribution (red-blue colormap, matplotlib)

### Clutches (`mecapy/clutches/`)
**Purpose:** friction couplings (Shigley Ch. 16). mm / N / N*mm / MPa convention.

- **`AxialFrictionInterface`** — shared annular friction math, parameterized by half-cone
  angle (90° = flat disc; cone torque scales by 1/sin(alpha))
  - Both theories always available as method pairs — no mode flag:
    `torque_uniform_wear(F)` / `torque_uniform_pressure(F)`, force/pressure inversions
  - **Vocabulary rule:** `p_max` args = uniform-WEAR theory (pressure peaks at the inner
    radius); `pressure` args = uniform-PRESSURE theory. Never conflate them.
  - Accepts explicit `mu` OR `lining="molded"` (lining supplies mu default and enables
    `pressure_safety_factor()` against the lining's p_max; explicit `mu` wins)
- **`DiscClutch`** — flat disc; `optimal_inner_diameter` (d = D/sqrt(3)); doubles as the
  full-annulus disc brake (re-exported as `mecapy.brakes.DiscBrake`)
- **`ConeClutch`** — cone_angle in (0, 45]; `is_self_holding` (tan alpha < mu), `face_width`
- **`CentrifugalClutch`** — engagement speed sqrt(S/(m*r_g)), torque grows with omega^2
- **`friction_data.py`** — `FRICTION_MATERIALS` lining table (mu dry/oil, p_max MPa,
  t_max °C) + `get_friction_material()`, mirroring the `*_data.py` accessor pattern

### Brakes (`mecapy/brakes/`)
**Purpose:** friction stopping elements. Reuses the clutch friction core.

- **`InternalShoeBrake` / `ExternalShoeBrake`** — pivoted long shoe (Shigley Eqs. 16-2..16-8)
  - `rotation="self_energizing"` / `"de_energizing"` selects the Mf sign branch
  - `moment_friction/moment_normal/actuating_force/torque(p_max)`, inversions
    `max_pressure_for_force()` / `torque_for_force()` (how the secondary shoe of a
    two-shoe brake is solved at the shared actuating force)
  - `is_self_locking` / `self_locking_margin` (MN/Mf, pressure-independent);
    `hinge_reactions()` in the Fig. 16-7 shoe frame
  - Validated against Shigley Ex. 16-2 (F ≈ 2.28 kN, total T ≈ 528 N*m)
- **`BandBrake`** — `tension_ratio` = e^(mu*phi), torque/tension/pressure round-trips,
  `band_stress()` + `band_safety_factor()` (needs `band_thickness`)
- **`CaliperDiscBrake`** — annular-sector pads, effective radius per theory
- **`DiscBrake`** — alias of `DiscClutch` (full-annulus disc brake IS a disc clutch)

### Couplings (`mecapy/couplings/`)
**Purpose:** shaft-to-shaft connections.

- **`FlangeCoupling`** — rigid: bolt shear (0.577*Sy), flange bearing, hub-flange shear,
  key shear/bearing checks; `torque_capacity(safety_factor=...)` returns the weakest-mode
  torque (skips checks whose geometry wasn't given)
- **`FlexibleCoupling`** — catalog-style ratings: torque/speed safety factors,
  `check_misalignment()` / `validate_misalignment()` (angular/parallel/axial),
  torsional windup (needs `torsional_stiffness`)

### Belts (`mecapy/belts/`)
**Purpose:** flexible mechanical elements — flat and V-belt drives (Shigley Ch. 17).
mm / N / W / MPa convention; belt/pulley speed in m/s, mass per length in kg/m.

- **`FlatBelt`** — open or crossed drive around two pulleys; wrap angles (Eq. 17-1),
  belt length (Eq. 17-2/17-3), capstan tension ratio `e^(mu*phi)` over the governing
  (smaller) wrap angle, centrifugal tension `Fc = m'V^2`, `tensions_for_power()` /
  `power()` round-trip at full capstan development, `initial_tension()` (Eq. 17-9)
  - mu/density/allowable_stress resolution mirrors `BandBrake`: explicit arg wins,
    then `belt_material` table row (`belts/belt_data.py`), then a friction default
  - Never define a method named `safety_factor` on these classes (would shadow the
    inherited Pa-based `MechaElement.safety_factor`); use `belt_stress_safety_factor()`
    and `power_safety_factor()` instead
- **`VBelt(FlatBelt)`** — groove wedging multiplies the effective friction (same move
  as `ConeClutch` on a flat disc): `effective_friction` = `mu/sin(groove_angle/2)` when
  `mu` is given, else Shigley's tabulated `V_BELT_EFFECTIVE_MU` (0.5123, Eq. 17-24)
  directly. `pitch_length()` / `center_distance_for_pitch_length()` round-trip
  (Eq. 17-16a/b), `designation` (e.g. "B90"), `belts_required()` (per-belt power
  supplied by the caller; full Kb/Kc/K1/K2 rating tables are not implemented)
- **`belt_data.py`** — `FLAT_BELT_MATERIALS` / `V_BELT_SECTIONS` tables + accessors,
  mirroring the `clutches/friction_data.py` pattern

### Chains (`mecapy/chains/`)
**Purpose:** ANSI roller chain drives (Shigley Ch. 17, Sec. 17-4). Same mm/N/W
convention; chain speed in m/s.

- **`RollerChain`** — sprocket pitch diameters (Eq. 17-28), chordal speed variation,
  chain length / center-distance round-trip (Eq. 17-34/17-35), `working_tension()` and
  `tensile_safety_factor()`, `rated_power()` = min(H1, H2) from the empirical
  pre-extreme-horsepower equations (Eq. 17-32 link-plate fatigue, Eq. 17-33
  roller-bushing wear — computed internally in inches/hp, converted once at the
  boundary with `utils.converters.hp_to_kw`); service factors and lubrication regime
  selection are not modeled
- **`chain_data.py`** — `ANSI_ROLLER_CHAINS` table (pitch, width, min tensile strength,
  mass per length, the Eq. 17-33 `kr` constant) + `STRAND_FACTORS` (Table 17-22) +
  `get_chain()` accessor

### Flywheels (`mecapy/wheels/flywheel.py`)
**Purpose:** rotational energy storage (Shigley Sec. 16-12). **SI units** (m, kg, Pa) —
extends the SI-era `Wheel`, which now has `moment_of_inertia` and `kinetic_energy(omega)`.

- Sizing: `Flywheel.required_inertia(Ue, Cs, omega_avg)` (staticmethod, works before a
  geometry exists), `energy_fluctuation()`, `coefficient_of_fluctuation()`, `speed_swing()`
- Stresses: full rotating annular/solid-disc `tangential_stress()` / `radial_stress()`
  (Shigley Eq. 3-55), `rim_hoop_stress()` quick check, `burst_safety_factor()`,
  `max_speed()` / `max_speed_rpm()`

### Thermal helpers (`mecapy/utils/thermal.py`)
**Purpose:** clutch/brake energy and temperature rise (Shigley Secs. 16-8/16-9). Pure SI.

- `stop_energy(I, w1, w2)`, `clutch_slip_energy(I1, I2, w1, w2)` (two-inertia engagement),
  `engagement_time(...)`, `temperature_rise(E, m, specific_heat= | material=)`,
  `cooling_time_constant()` + `newton_cooling_temperature()` (Newton cooling decay),
  `interface_power(T, w)`, `pv_value(p, V)`
- `MATERIALS` entries now include `specific_heat` (J/(kg*K)) for the material= lookup
- Bridge from geometry modules: torque N*mm -> N*m is /1e3 (done once in clutch `.power()`)

### Units convention

Material properties are SI (Pa, m, kg/m^3). Geometry inputs on individual elements
(gears, shafts, bolts, bearings, welds) are documented in **mm**, with results such as
`Shaft.torsional_stress` returning MPa for mm/N*mm inputs. `mecapy/utils/converters.py`
holds the mm/m, kPa/Pa/MPa, N/kgf, in/mm, lbf/N, psi/MPa and hp/kW conversions used to
bridge between them. `Beam` (SymPy-backed) works in "consistent units" and stays symbolic
where possible rather than committing to one system.

### The `Beam` idiom (SymPy-backed elements)

`mecapy/beams/beam.py` wraps `sympy.physics.continuum_mechanics.beam.Beam`. It is the
model other "solve"-style elements should follow if extended: builder methods
(`add_support`, `add_load`, `add_point_load`, `add_moment`, `add_distributed_load`)
return `self` for chaining and accumulate reaction symbols; a lazy `solve()` /
`_ensure_solved()` guard defers the actual symbolic solve until a result (`reactions`,
`shear_force`, `bending_moment`, `deflection`, `max_bending_moment`, `bending_stress`, ...)
is requested. `mecapy/beams/calculations.py` holds separate standalone closed-form
helpers (cantilever deflection, M*c/I bending stress, V/A shear stress) that don't
require building a full `Beam`.

### Gear Subsystem Deep Dive (`mecapy/gears/`)

This is the largest and most structured subsystem. The design prioritizes extensibility:
gear types, mesh compatibility, and rating methods should be independent but coordinated.

#### Core Design Principles

**1. Unit conversion at the boundary:**
- Constructor accepts EITHER `module` (mm, SI) OR `diametral_pitch` (teeth/inch)
- Exactly one must be given; the other is rejected with a clear error
- `diametral_pitch` is converted to `module` internally (e.g., 12 DP → 2.117 mm)
- All internal geometry, mesh checks, and rating use metric-only

**2. Pair-wise geometry is a method, not a property:**
- Properties like `pitch_diameter`, `outside_diameter` belong to a single gear
- Anything that depends on a mating gear is a `*_with(other)` method:
  ```python
  spur1 = SpurGear(teeth=20, module=2)
  spur2 = SpurGear(teeth=60, module=2)
  center_dist = spur1.center_distance_with(spur2)  # method, not property
  contact_ratio = spur1.contact_ratio_with(spur2)
  ```
- This forces you to think about mesh compatibility upfront

**3. Single source of truth for mesh rules (`Transmission._check_mesh()`):**
- `Transmission._check_mesh(driver, driven)` validates:
  - Module and pressure angle match (or compatible)
  - Gear types are compatible (e.g., no spur + bevel unless bridged)
  - Helical hands are opposite for external gears
  - Rack/Worm terminal rules are obeyed
- Other pair methods (`contact_ratio_with()`, `PlanetaryGearSet.__init__()`) reuse this
- Reduces the risk of silently accepting bad meshes

#### Type Hierarchy

```
Gear (gear.py) — base, standard involute geometry
├── CylindricalGear (cylindrical.py) — parallel axes
│   ├── SpurGear — no helix
│   ├── HelicalGear — helix_angle + hand ("right"/"left")
│   └── HerringboneGear(HelicalGear) — hand = None (thrust cancels)
├── BevelGear (bevel.py) — intersecting axes (cone mesh)
│   └── Methods like `cone_angle_with(mate)` for pair-wise geometry
└── WormWheel (worm.py) — special: meshes only with Worm

Worm (worm.py) — NOT a Gear subclass
  • Has `starts` and `lead`, not teeth/module
  • Linear velocity is different from rotational velocity
  • Pair-wise methods like `velocity_ratio_with(wheel)` on either side

Rack (rack.py) — NOT a Gear subclass
  • Infinite pitch radius (linearized gear)
  • Terminal in a transmission (can only be driven, not drive)
  • Methods: `center_distance_with(gear)`, `contact_ratio_with(gear)`

PlanetaryGearSet (planetary.py) — composite
  • Sun + planet(s) + ring, all three rotating on the same axis
  • Willis train-value equation solves speed/torque relationships
  • NOT a `Transmission` stage (stage = two-element mesh, this = three-member)
  • Build manually: `PlanetaryGearSet(sun=20, planet=30, ring=80)`

Transmission (transmission.py) — orchestrates meshes
  • Chains 2+ elements (gears, rack, worm) into a kinematic system
  • Computes overall ratio, output speed/torque, pitch-line velocity
  • Calls `_check_mesh()` to validate each stage
  • Does NOT include planetary sets (compose manually)
```

#### Rating Strategies

**AGMA 2101-D04** (Cylindrical gears only: Spur, Helical, Herringbone, pinion-on-rack)
- Files: `agma.py`, `agma_data.py`
- Two equations:
  - **Bending:** Lewis stress with geometry factor J, K factors (load, life, reliability)
  - **Pitting:** Hertzian contact stress with surface finish, hardness factors
- Data tables digitized from AGMA 2101-D04:
  - Lewis geometry factor Y for various tooth counts and pressure angles
  - J factor for different gear ratios
  - Material allowables for various hardnesses
  - Cma load distribution factors
  - Reliability factors (1.0 = 99% survival, down to 0.5 for 99.99%)
- Accepts explicit overrides: `AGMARating(..., YJ=0.48, Cma=1.2)` to bypass table lookup

**Simplified methods** (Bevel & Worm — educational approximations):
- `BevelGear.lewis_bending_stress()` — Lewis equation with cone-angle correction
- `Worm.permissible_load()` — Buckingham approximation for wear & bending
- Documented as **not full AGMA 2003/6034** (the real standards)
- Suitable for design exploration; full rating requires domain expertise

#### Common Pitfalls & How to Avoid Them

| Problem | Cause | Solution |
|---------|-------|----------|
| "Module mismatch" error | Used `module=2` in one gear, `diametral_pitch=12` in another | Always use the same unit — either metric or US customary across a mesh |
| "Rack must be terminal" | Tried to drive a Rack as if it were a gear | Rack can only be the last (output) stage |
| "Worm/WormWheel only mesh together" | Mixed a Worm with a SpurGear | Check gear types match in `Transmission` |
| Pair method returns 0 | Two gears have incompatible pressure angles | Ensure both use same pressure angle (default 20°) |
| Unexpected AGMA rating | Material allowables look wrong | Check `agma_data.py` tables; override with explicit `allowable_stress_bending=...` if needed |

### Validation Pattern

Constructors validate eagerly and raise `ValueError` with a specific message for every
non-physical input (zero/negative dimensions, out-of-range angles, incompatible
mesh pairs, failed planetary assembly/geometric conditions). There is no silent clamping
or default-substitution for bad input anywhere in the codebase.

Example (from `PowerScrew.__init__`):
```python
if major_diameter <= 0:
    raise ValueError("Major diameter must be strictly positive")
if pitch >= major_diameter:
    raise ValueError("Pitch too large: root diameter would be non-positive")
```

This pattern applies everywhere — fail fast and loud, with a message that tells the user WHY.

## How to Extend the Codebase

### Adding a New Mechanical Element

**Pattern:** inherit from `MechaElement`, implement geometry + stress calculation.

1. **Create a new file** in the appropriate subsystem (e.g., `mecapy/bearings/spherical.py`)
2. **Inherit from `MechaElement`:**
   ```python
   from ..base import MechaElement
   
   class SphericalBearing(MechaElement):
       def __init__(self, bore_diameter, material="steel", name=None):
           super().__init__(name=name, material=material)
           if bore_diameter <= 0:
               raise ValueError("Bore diameter must be strictly positive")
           self.bore_diameter = bore_diameter
   ```
3. **Add geometry properties** using `@property`:
   ```python
   @property
   def raceway_stress_area(self):
       """Contact area in mm^2."""
       return math.pi * self.bore_diameter * self.height
   ```
4. **Implement stress calculation** (optional override):
   ```python
   def contact_stress(self, load_force):
       """Hertzian contact stress for radial load."""
       area = self.raceway_stress_area
       return self.calculate_stress(load_force, area)
   ```
   Or use the inherited `calculate_stress(force, area)` directly.
5. **Add tests** in `tests/test_<subsystem>.py`
6. **Update imports** in `mecapy/__init__.py` and the subsystem's `__init__.py`

### Adding a New Gear Type

**Pattern:** inherit from `CylindricalGear` or `Gear`, implement pair-wise methods.

1. **Inherit from the right base:**
   - If parallel axes with involute teeth → inherit `CylindricalGear`
   - If custom geometry → inherit `Gear` directly
2. **Add constructor validation:**
   ```python
   class StraightBevelGear(BevelGear):
       def __init__(self, teeth, module, cone_angle, ...):
           if not 0 < cone_angle < 90:
               raise ValueError("Cone angle must be in (0, 90) degrees")
   ```
3. **Implement pair-wise geometry:**
   ```python
   def contact_ratio_with(self, other):
       """Override or inherit from base."""
       # Validate compatibility
       Transmission._check_mesh(self, other)
       # Compute pair-wise value
       return ...
   ```
4. **Hook into mesh validation** — update `Transmission._check_mesh()` if new rules apply
5. **Add rating method** if you're adding a new gear family:
   - Cylindrical family → use AGMA 2101-D04 (already in `agma.py`)
   - Bevel/Worm → add a simplified method in the gear class itself

### Adding a Stress Rating Method

**Pattern:** class method or standalone function in a dedicated module.

Example: `BevelGear.lewis_bending_stress()`

```python
def lewis_bending_stress(self, power_kw, speed_rpm, overload_factor=1.0):
    """Lewis equation for bending stress (educational approximation)."""
    # Convert power to force
    force = self.force_from_power(power_kw, speed_rpm)
    # Apply Lewis geometry factor (from data table or formula)
    lewisfactor = self._lewis_y_factor()
    stress = (force * self.face_width) / (self.module * lewisfactor)
    return stress * overload_factor
```

For AGMA-style ratings with multiple factors, create a data module (`agma_data.py`) with:
- Digitized tables (curve-fit or hardcoded grids)
- Factor lookup functions
- Reference documentation

### Common Code Patterns to Follow

| Goal | Pattern |
|------|---------|
| Make a value computed but look like an attribute | Use `@property` |
| Compute something that depends on a mating element | Use `method_name_with(other)` |
| Validate a single bad input | Raise `ValueError("message")` in `__init__` |
| Share data across many elements | Put it in `mecapy/utils/constants.py` or a subsystem `_data.py` |
| Solve a system of equations symbolically | Use `SymPy` (see `Beam` for the pattern) |
| Convert between unit systems | Use functions in `mecapy/utils/converters.py` |
| Document an approximation or limitation | Add a docstring note or `# Docstring reference:` comment |

## Testing Strategy

**Unit tests** (`tests/test_*.py`):
- Test constructors with valid/invalid inputs
- Test individual property calculations against known examples
- Mock external data (threads, materials) if needed

**Integration tests:**
- Build a gearbox (`Transmission`) and verify kinematic output matches expected ratio
- Calculate full AGMA rating and compare against published examples (e.g., Shigley textbook)
- Test that a bolted joint under eccentric load distributes correctly

**Fixtures** (`conftest.py`):
- Define standard test gears, shafts, bolts so tests reuse them

**Run with coverage:**
```bash
pytest --cov=mecapy --cov-report=html
# Open htmlcov/index.html to see which lines are tested
```

## Common Questions

**Q: Where do I find the stress area of an M10 bolt?**  
A: `Bolt("M10").stress_area` — it's looked up from `mecapy/bolts/thread_data.py`.

**Q: How do I set up a gearbox and check the rating?**  
A: Use `Transmission` to chain gears, then `AGMARating` on each stage:
```python
from mecapy.gears import SpurGear, Transmission
from mecapy.gears.agma import AGMARating

pinion = SpurGear(teeth=20, module=2)
gear = SpurGear(teeth=60, module=2)
trans = Transmission(pinion, gear)

rating = AGMARating(pinion, gear, power_kw=10, speed_rpm=1800)
print(f"Bending safety factor: {rating.safety_factor_bending}")
print(f"Contact safety factor: {rating.safety_factor_contact}")
```

**Q: What's the difference between a Rack and a normal gear?**  
A: A Rack has infinite pitch radius — it's a linearized gear that meshes with a rotational gear to convert between rotation and translation. It can only be driven (terminal in a `Transmission`).

**Q: Can I mesh a Worm with a SpurGear?**  
A: No. `Transmission._check_mesh()` will raise an error. Worms mesh only with WormWheels. This constraint is enforced everywhere, not silently ignored.

**Q: How do I add custom material properties?**  
A: `mecapy.materials.add_custom_material("my_alloy", yield_strength=450e6, elastic_modulus=210e9, ...)` — then use `material="my_alloy"` in any element constructor.
