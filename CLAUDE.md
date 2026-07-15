# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
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
(e.g. `python examples/gear_design.py`) need `PYTHONPATH=.` unless `pip install -e .`
has been run first.

flake8 config: max line length 100, ignores E203/W503 (`setup.cfg`). mypy: Python 3.8 target,
`disallow_untyped_defs = False` — the codebase does not use type annotations, relying on
Google-style docstrings instead.

## Architecture

Every mechanical component inherits from a single base class, `MechaElement`
(`mecapy/base.py`), which provides shared material access and the fundamental
stress/safety-factor calculations:

```
MechaElement                # material_properties, calculate_stress(F, A), safety_factor(stress)
├── Beam (beams/)           # SymPy-backed: reactions, shear, moment, deflection
├── Shaft (shafts/)         # + torsional_stress()
├── Gear (gears/)           # base gear geometry; see gear type hierarchy below
├── Wheel (wheels/)
├── Bearing (bearings/)
├── Bolt (bolts/)
└── Weld (welds/)
```

Because every element shares `MechaElement`, any of them can compute an axial stress
and a safety factor against yielding via the same two inherited methods — there is no
per-element stress/safety duplication.

Materials are looked up by name (`"steel"`, `"aluminum"`, `"copper"`, `"cast_iron"`,
`"bronze"`) from a database in `mecapy/utils/constants.py` (`MATERIALS` dict), accessed
through `mecapy/materials.py` (`get_material_properties`, `get_available_materials`,
`add_custom_material`). All material properties are strict SI (Pa, kg/m^3).

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

### Gear subsystem (`mecapy/gears/`)

This is the largest and most structured subsystem. `Gear` (`gear.py`) is the common base
(backward-compatible, directly constructible) with full standard full-depth geometry
(pitch/base/outside/root diameter, circular pitch, addendum/dedendum). Every gear
constructor accepts either `module` (mm, primary/SI) or `diametral_pitch` (teeth/inch,
US customary) — exactly one must be given; `diametral_pitch` is converted to `module`
internally so all downstream geometry and rating code is metric-only.

```
Gear (gear.py)
├── CylindricalGear (cylindrical.py)   # normal module/pressure angle, transverse geometry
│   ├── SpurGear
│   ├── HelicalGear                    # requires helix_angle + hand ("right"/"left")
│   └── HerringboneGear(HelicalGear)   # hand forced None; thrust cancels
├── BevelGear (bevel.py)               # pair-wise cone geometry (*_with(mate) methods)
└── WormWheel (worm.py)                # meshes only with Worm, not Gear-derived on its own axis

Worm (worm.py)                          # NOT a Gear subclass — has starts/lead, not teeth/module in the same sense
Rack (rack.py)                          # NOT a Gear subclass — infinite pitch radius, no teeth count
PlanetaryGearSet (planetary.py)         # composite of sun/planet/ring Gear objects or plain ints
Transmission (transmission.py)          # chains 2+ meshes; owns the mesh-compatibility rules
```

Key structural points for extending this subsystem:

- **Pair-wise geometry is a method, not a property.** Anything that depends on a mating
  gear (contact ratio, center distance, bevel cone angle, worm ratio) is a
  `*_with(other)` method on one of the two gears, not a standalone property — because
  the value has no meaning without the mate.
- **`Transmission._check_mesh(driver, driven)`** (in `transmission.py`) is the single
  source of truth for whether two elements can mesh (matching module and pressure angle,
  compatible types, opposite hands for external helical gears, rack/worm must be
  terminal). Other pair methods (`contact_ratio_with`, `PlanetaryGearSet` construction)
  import and reuse this function rather than re-implementing compatibility checks.
- **AGMA rating (`agma.py` + `agma_data.py`)** implements the AGMA 2101-D04 metric
  bending/pitting equations (Shigley's *Mechanical Engineering Design* ch. 14 is the
  numeric reference) for the cylindrical family only (spur/helical/herringbone,
  including a pinion-on-rack). Bevel and worm gears intentionally get simplified
  Lewis-equation / Buckingham-style checks instead (`BevelGear.lewis_bending_stress`,
  `Worm.permissible_load`) — they are documented as educational approximations, not
  full AGMA 2003/6034 ratings. `agma_data.py` holds digitized/curve-fit tables (Lewis Y,
  J geometry-factor grids, Cma coefficients, reliability factors, gear-steel allowables)
  that `agma.py`'s factor functions and the `AGMARating` class consume; any rating
  accepts explicit overrides (e.g. `YJ=`) to bypass a table lookup.
- **`PlanetaryGearSet` is deliberately not a `Transmission` stage** — a stage is a
  two-element mesh, while a planetary set is a three-member (sun/planet/ring) composite
  solved via the Willis train-value equation. Compose their ratios manually when both
  appear in the same gearbox.

### Validation pattern

Constructors validate eagerly and raise `ValueError` with a specific message for every
non-physical input (zero/negative dimensions, out-of-range angles, incompatible
mesh pairs, failed planetary assembly/geometric conditions). There is no silent clamping
or default-substitution for bad input anywhere in the codebase.
