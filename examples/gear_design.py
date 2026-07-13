"""Example: Gear design using MecaPy."""

from mecapy.gears import Gear
from mecapy.materials import get_material_properties


def design_spur_gear():
    """Design a spur gear."""
    # Create a spur gear
    gear = Gear(teeth=20, module=2.5, material="steel")

    # Get material properties
    steel = get_material_properties("steel")

    # Calculate pitch diameter
    pitch_diameter = gear.teeth * gear.module

    print("Spur Gear Design")
    print("=" * 40)
    print(f"Number of teeth: {gear.teeth}")
    print(f"Module: {gear.module} mm")
    print(f"Material: {gear.material}")
    print(f"Pitch Diameter: {pitch_diameter} mm")
    print()
    print("Material Properties:")
    print(f"  Density: {steel['density']} kg/m^3")
    print(f"  Elastic Modulus: {steel['elastic_modulus']/1e9:.1f} GPa")
    print(f"  Yield Strength: {steel['yield_strength']/1e6:.1f} MPa")
    print()


if __name__ == "__main__":
    design_spur_gear()
