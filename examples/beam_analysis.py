"""Example: Cantilever beam analysis using MecaPy."""

from mecapy.beams import Beam
from mecapy.materials import get_material_properties
from mecapy.utils import converters


def analyze_cantilever_beam():
    """Analyze a cantilever beam."""
    # Create a cantilever beam
    beam = Beam(length=3.0, material="steel", section={"width": 0.1, "height": 0.2})

    # Get material properties
    steel = get_material_properties("steel")

    print("Cantilever Beam Analysis")
    print("=" * 40)
    print(f"Beam length: {beam.length} m")
    print(f"Material: {beam.material}")
    print(f"Cross-section: {beam.section}")
    print()
    print("Material Properties:")
    print(f"  Density: {steel['density']} kg/m^3")
    print(f"  Elastic Modulus: {steel['elastic_modulus']/1e9:.1f} GPa")
    print(f"  Yield Strength: {steel['yield_strength']/1e6:.1f} MPa")
    print()


if __name__ == "__main__":
    analyze_cantilever_beam()
