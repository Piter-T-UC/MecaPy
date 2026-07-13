# MecaPy - Python Library for Mechanical Engineering Calculations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive Python library for mechanical engineering calculations and design analysis.

## Features

MecaPy provides tools for analyzing and designing various mechanical components:

- **Beams** - Analysis of beams under various loading conditions
- **Wheels** - Design and analysis of wheels
- **Gears** - Gear design and transmission analysis
- **Bearings** - Bearing selection and life prediction
- **Bolts** - Bolt stress and fastener analysis
- **Welds** - Weld design and fatigue analysis
- **Shafts** - Shaft design and deflection analysis

## Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)

### From Source

```bash
git clone https://github.com/piter-t-uc/mecapy.git
cd mecapy
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Architecture

Every component inherits from a common base class, `MechaElement`, which
provides shared access to material properties and the fundamental
stress/safety-factor calculations:

```
MechaElement                # material, calculate_stress(), safety_factor()
├── Beam                    # SymPy-backed: reactions, shear, moment, deflection
├── Shaft                   # + torsional_stress()
├── Gear                    # + pitch_diameter
├── Wheel
├── Bearing
├── Bolt
└── Weld
```

Because they all share `MechaElement`, any element can compute an axial
stress and its safety factor against yielding:

```python
from mecapy.bolts import Bolt

bolt = Bolt(diameter=10.0, length=50.0, material="steel")
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

### Designing a Gear

```python
from mecapy.gears import Gear

# Create a spur gear
gear = Gear(teeth=20, module=2.5, material="steel")
print(f"Pitch Diameter: {gear.pitch_diameter} mm")
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

Run the test suite with pytest:

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

Get available materials:

```python
from mecapy.materials import get_available_materials

materials = get_available_materials()
print(materials)  # ['steel', 'aluminum', 'copper', 'cast_iron']
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
