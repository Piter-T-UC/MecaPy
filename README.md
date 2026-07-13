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

## Quick Start

### Creating and Analyzing a Beam

```python
from mecapy.beams import Beam
from mecapy.materials import get_material_properties

# Create a steel beam
beam = Beam(length=5.0, material="steel")

# Get material properties
steel = get_material_properties("steel")
print(f"Elastic Modulus: {steel['elastic_modulus']/1e9:.1f} GPa")
```

### Designing a Gear

```python
from mecapy.gears import Gear

# Create a spur gear
gear = Gear(teeth=20, module=2.5, material="steel")
pitch_diameter = gear.teeth * gear.module
print(f"Pitch Diameter: {pitch_diameter} mm")
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
