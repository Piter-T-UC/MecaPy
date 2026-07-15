Quick Start
===========

Architecture
------------

Every component inherits from a common base class,
:class:`~mecapy.base.MechaElement`, which provides shared access to material
properties and the fundamental stress and safety-factor calculations. This
means any element can compute a direct stress and its safety factor against
yielding.

::

    from mecapy.bolts import Bolt

    bolt = Bolt(size="M10", length=50.0, material="steel")
    stress = bolt.calculate_stress(force=5000, area=80)  # -> 62.5 MPa
    print(bolt.safety_factor(stress * 1e6))

Analyzing a Beam
----------------

The :class:`~mecapy.beams.Beam` class is backed by SymPy's
continuum-mechanics beam. Add supports and loads, then read off symbolic
reactions, bending moments and deflections.

::

    from mecapy.beams import Beam

    # 6 m steel beam; E from the material database, I supplied explicitly
    beam = Beam(length=6.0, material="steel", second_moment=8.0e-6)

    beam.add_support(0, "pin").add_support(6, "roller")
    beam.add_point_load(-2000, 3)  # 2 kN downward at midspan

    print(beam.reactions)                        # {R_0: 1000, R_6: 1000}
    location, moment = beam.max_bending_moment()  # (3, 3000) N*m
    stress = beam.bending_stress(distance_to_fiber=0.1)
    print(f"Safety factor: {beam.safety_factor(float(stress)):.1f}")

Designing a Gear
----------------

::

    from mecapy.gears import Gear

    gear = Gear(teeth=20, module=2.5, material="steel")
    print(f"Pitch Diameter: {gear.pitch_diameter} mm")

Torsion on a Shaft
------------------

::

    from mecapy.shafts import Shaft

    shaft = Shaft(diameter=25.0, length=500.0, material="steel")
    print(f"{shaft.torsional_stress(150_000):.1f} MPa")  # torque in N*mm

Accessing Material Properties
-----------------------------

::

    from mecapy.materials import get_material_properties

    steel = get_material_properties("steel")
    print(f"Young's modulus: {steel['elastic_modulus'] / 1e9:.1f} GPa")

Next Steps
----------

- Explore the :doc:`../modules/index` for detailed API documentation
- Check out :doc:`../examples/index` for more detailed examples
