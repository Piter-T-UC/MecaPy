Beam Analysis Example
=====================

This example demonstrates how to analyze a cantilever beam using MecaPy.

::

    from mecapy.beams import Beam
    from mecapy.materials import get_material_properties
    from mecapy.utils import converters

    # Create a cantilever beam
    beam = Beam(
        length=3.0,  # 3 meters
        material="steel",
        section={"width": 0.1, "height": 0.2}  # 0.1m x 0.2m
    )

    # Get material properties
    steel = get_material_properties("steel")
    print(f"Material: {beam.material}")
    print(f"Density: {steel['density']} kg/m^3")
    print(f"Elastic Modulus: {steel['elastic_modulus']/1e9:.1f} GPa")

    # Example calculations would go here
    # (Implementation to be completed based on engineering requirements)

Expected Output::

    Material: steel
    Density: 7850 kg/m^3
    Elastic Modulus: 210.0 GPa
