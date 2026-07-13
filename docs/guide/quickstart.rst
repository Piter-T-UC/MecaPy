Quick Start
===========

Basic Usage
-----------

Creating a Beam
~~~~~~~~~~~~~~~

::

    from mecapy.beams import Beam

    # Create a simply supported steel beam
    beam = Beam(length=5.0, material="steel")
    print(beam)

Creating a Gear
~~~~~~~~~~~~~~~

::

    from mecapy.gears import Gear

    # Create a spur gear
    gear = Gear(teeth=20, module=2.5, material="steel")
    print(gear)

Creating a Shaft
~~~~~~~~~~~~~~~~

::

    from mecapy.shafts import Shaft

    # Create a transmission shaft
    shaft = Shaft(diameter=25.0, length=500.0, material="steel")
    print(shaft)

Accessing Material Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from mecapy.materials import get_material_properties

    # Get properties of steel
    steel_props = get_material_properties("steel")
    print(f"Steel density: {steel_props['density']} kg/m^3")
    print(f"Young's modulus: {steel_props['elastic_modulus']} Pa")

Unit Conversions
~~~~~~~~~~~~~~~~

::

    from mecapy.utils.converters import mm_to_m, mpa_to_pa

    # Convert units
    length_m = mm_to_m(500)  # 500 mm to meters
    stress_pa = mpa_to_pa(250)  # 250 MPa to pascals

Next Steps
----------

- Explore the :doc:`../modules/index` for detailed API documentation
- Check out :doc:`../examples/index` for more detailed examples
- Read the contributing guide to help improve MecaPy
