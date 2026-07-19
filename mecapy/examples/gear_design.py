"""Example: Gear design using MecaPy."""

from mecapy.gears import SpurGear, Transmission
from mecapy.materials import get_material_properties


def design_spur_gear():
    """Design a spur gear and inspect its geometry."""
    # Create a spur gear
    gear = SpurGear(teeth=20, module=2.5, material="steel")

    # Get material properties
    steel = get_material_properties("steel")

    print("Spur Gear Design")
    print("=" * 40)
    print(f"Number of teeth: {gear.teeth}")
    print(f"Module: {gear.module} mm")
    print(f"Material: {gear.material}")
    print(f"Pitch Diameter: {gear.pitch_diameter} mm")
    print(f"Base Diameter: {gear.base_diameter:.2f} mm")
    print(f"Outside Diameter: {gear.outside_diameter} mm")
    print(f"Root Diameter: {gear.root_diameter} mm")
    print(f"Circular Pitch: {gear.circular_pitch:.3f} mm")
    print(f"Min teeth (no undercut): {gear.min_teeth_no_undercut}")
    print()
    print("Material Properties:")
    print(f"  Density: {steel['density']} kg/m^3")
    print(f"  Elastic Modulus: {steel['elastic_modulus']/1e9:.1f} GPa")
    print(f"  Yield Strength: {steel['yield_strength']/1e6:.1f} MPa")
    print()


def design_compound_train():
    """Two-stage compound train: 17->68 then 18->54 (ratio 12)."""
    transmission = (
        Transmission(name="two-stage reducer")
        .add_stage(SpurGear(17, module=2.0), SpurGear(68, module=2.0))
        .add_stage(SpurGear(18, module=2.0), SpurGear(54, module=2.0))
    )

    input_speed = 1200.0  # rpm
    input_torque = 15.0  # N*m

    print("Compound Gear Train")
    print("=" * 40)
    print(f"Overall ratio: {transmission.overall_ratio:.1f}")
    print(f"Input speed: {input_speed:.0f} rpm")
    print(f"Output speed: {transmission.output_speed(input_speed):.1f} rpm")
    print(f"Input torque: {input_torque:.1f} N*m")
    print(f"Output torque: {transmission.output_torque(input_torque):.1f} N*m")
    for i, stage in enumerate(transmission.speeds(input_speed)):
        print(f"  Stage {i + 1}: {stage['driver_rpm']:.1f} rpm -> "
              f"{stage['driven_rpm']:.1f} rpm "
              f"(center distance {transmission.center_distance(i):.1f} mm)")
    print()


if __name__ == "__main__":
    design_spur_gear()
    design_compound_train()
