# MecaPy - Python Library for Mechanical Engineering Calculations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive Python library for mechanical engineering calculations and design analysis.

## Features

MecaPy provides tools for analyzing and designing various mechanical components:

- **Beams** - SymPy-backed analysis of beams under various loading conditions
- **Wheels & Flywheels** - Rotating members: inertia, kinetic energy, flywheel sizing and burst speed
- **Gears** - Full gear family (spur, helical, herringbone, bevel, worm, rack, planetary),
  transmissions, profile shift, and AGMA 2101-D04 bending/pitting rating
- **Bearings** - Rolling-contact life (Shigley ch. 11 + ISO 281 modified rating life,
  ISO 76 static rating), hydrodynamic journal and tapered-land thrust bearings (ch. 12),
  and boundary-lubricated bushings rated on PV
- **Bolts** - ISO metric bolt stress plus multi-bolt joint (`BoltedUnion`) load distribution
- **Welds** - Weld group analysis by the "weld as a line" method, sizing and stress plots
- **Shafts & Power Screws** - Torsion, deflection, and lead-screw torque/efficiency/self-locking
- **Clutches & Brakes** - Disc, cone and centrifugal clutches; shoe, band and caliper disc brakes
  (Shigley Ch. 16, uniform-pressure and uniform-wear theories)
- **Couplings** - Rigid flange couplings and catalog-style flexible couplings
- **Thermal helpers** - Clutch/brake stop energy, temperature rise and Newton cooling

## Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)

### From Source

The Python project lives in the `mecapy/` subdirectory of the repository.
All development commands are run from there:

```bash
git clone https://github.com/piter-t-uc/mecapy.git
cd mecapy/mecapy
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"          # pytest, coverage, black, flake8, mypy
pip install -e ".[dev,viz]"      # + matplotlib, for the plot_* methods
pip install -e ".[docs]"         # sphinx, for building the documentation
```

## Architecture

Every component inherits from a common base class, `MechaElement`, which
provides shared access to material properties and the fundamental
stress/safety-factor calculations:

```
MechaElement                              # material, calculate_stress(), safety_factor()
├── Beam                                  # SymPy-backed: reactions, shear, moment, deflection
├── Shaft                                 # + torsional_stress()
│   └── PowerScrew                        # lead screw: thread stresses, buckling, efficiency
├── Gear                                  # spur / helical / herringbone / bevel / worm / rack
├── Wheel                                 # pulleys, sprockets; inertia, kinetic energy
│   └── Flywheel                          # energy fluctuation sizing, rotating-disc stresses
├── Bearing                               # rolling contact: ch.11 life + ISO 281/76
├── JournalBearing                        # hydrodynamic plain journal (Shigley ch.12)
├── PlainBearing                          # boundary-lubricated bushing, PV limits
├── ThrustBearing                         # fixed-incline (tapered-land) thrust pad
├── Bolt                                  # ISO metric: tension, preload, stiffness
│   └── BoltedUnion                       # joint: load distribution, separation, efficiency
├── AxialFrictionInterface                # annular friction math (both wear theories)
│   ├── DiscClutch                        # flat disc (doubles as DiscBrake)
│   └── ConeClutch                        # wedging cone
├── CentrifugalClutch                     # spring-retained shoes, engagement speed
├── InternalShoeBrake / ExternalShoeBrake # pivoted long shoe (Shigley 16-2..16-8)
├── BandBrake                             # e^(mu*phi) tension ratio
├── CaliperDiscBrake                      # annular-sector pads
├── FlangeCoupling / FlexibleCoupling     # rigid and catalog-style shaft couplings
└── Weld                                  # weld bead stresses & material
    └── WeldedUnion                       # weld group: combined loading, sizing, plots
```

The gear subsystem also provides composite elements: `Transmission` chains gear
meshes into a kinematic system (ratios, output torque/power), and
`PlanetaryGearSet` solves sun/planet/ring trains via the Willis equation.

Because they all share `MechaElement`, any element can compute an axial
stress and its safety factor against yielding:

```python
from mecapy.bolts import Bolt

bolt = Bolt(size="M10", length=50.0, material="steel")
stress = bolt.calculate_stress(force=5000, area=80)   # N / mm^2 -> 62.5 MPa
print(bolt.safety_factor(stress * 1e6))               # vs. steel yield strength
```

## Quick Start

### Analyzing a Beam (powered by SymPy)

The `Beam` class wraps SymPy's continuum-mechanics beam, so you can add
supports and loads and get symbolic reactions, bending moments and
deflections:

```python
from mecapy.beams import Beam

# 6 m steel beam; E comes from the material database, I is supplied
beam = Beam(length=6.0, material="steel", second_moment=8.0e-6)

# Pin + roller supports and a 2 kN central point load
beam.add_support(0, "pin").add_support(6, "roller")
beam.add_point_load(-2000, 3)

print(beam.reactions)                       # {R_0: 1000, R_6: 1000}
location, moment = beam.max_bending_moment() # (3, 3000) N*m
stress = beam.bending_stress(distance_to_fiber=0.1)
print(f"{float(stress)/1e6:.1f} MPa, SF = {beam.safety_factor(float(stress)):.1f}")
```

### Designing a Gearbox

```python
from mecapy.gears import SpurGear, Transmission
from mecapy.gears.agma import AGMARating

pinion = SpurGear(teeth=20, module=2.5, material="steel", face_width=25)
gear = SpurGear(teeth=60, module=2.5, material="steel", face_width=25)
print(f"Pitch Diameter: {pinion.pitch_diameter} mm")
print(f"Center Distance: {pinion.center_distance_with(gear)} mm")

trans = Transmission().add_stage(pinion, gear)   # validates the mesh
print(f"Ratio: {trans.overall_ratio}")

rating = AGMARating(pinion, gear, power_kw=7.5, pinion_speed_rpm=1800,
                    hardness_HB=350)
print(f"Bending SF: {rating.SF_pinion:.2f}, Contact SF: {rating.SH:.2f}")
print(rating.summary())                          # full factor-by-factor report
```

### Sizing a Clutch

```python
from mecapy.clutches import DiscClutch

clutch = DiscClutch(outer_diameter=250, inner_diameter=150,
                    n_faces=2, lining="molded")  # lining supplies mu and p_max
torque = clutch.torque_uniform_wear(actuating_force=5000)   # N*mm
print(f"Torque capacity: {torque/1000:.1f} N*m")
```

### Torsion on a Shaft

```python
from mecapy.shafts import Shaft

shaft = Shaft(diameter=25.0, length=500.0, material="steel")
print(f"{shaft.torsional_stress(150_000):.1f} MPa")  # torque in N*mm
```

### Unit Conversions

```python
from mecapy.utils.converters import mm_to_m, mpa_to_pa

# Convert units
length_m = mm_to_m(500)  # 500 mm to meters
stress_pa = mpa_to_pa(250)  # 250 MPa to pascals
```

## Documentation

Full documentation is available at [ReadTheDocs](https://mecapy.readthedocs.io/) (coming soon)

For local documentation:

```bash
cd docs
make html
# Open _build/html/index.html in your browser
```

## Testing

Run the test suite with pytest (from the `MecaPy/` directory):

```bash
pytest
```

Generate coverage reports:

```bash
pytest --cov=mecapy --cov-report=html
```

## Available Materials

MecaPy includes a material database with properties for common engineering materials:

- Steel
- Aluminum
- Copper
- Cast Iron
- Bronze

Custom materials can be registered at runtime with
`mecapy.materials.add_custom_material()`. Friction lining data for clutches and
brakes lives in a separate table (`mecapy.clutches.friction_data`).

Get available materials:

```python
from mecapy.materials import get_available_materials

materials = get_available_materials()
print(materials)  # ['steel', 'aluminum', 'copper', 'cast_iron', 'bronze']
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Piter-T-UC**

## Acknowledgments

- Inspired by engineering design principles and standards
- Built with Python and scientific computing libraries

## Project Status

This project is currently in alpha development. Features and APIs may change.

## Roadmap

- [ ] Complete beam deflection calculations
- [ ] Add stress concentration factors
- [ ] Implement fatigue analysis
- [ ] Add FEA integration
- [ ] Create GUI for design calculations
- [ ] Publish to PyPI

## Support

For issues and questions:
- GitHub Issues: [GitHub Repository](https://github.com/piter-t-uc/mecapy/issues)
- Email: pedrito00.taboada@gmail.com
