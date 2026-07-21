"""Example: beam analysis with MecaPy.

Covers everything on the 2D Beam (supports, every load type including
q = f(x) function loads, reactions, shear/moment/slope/deflection,
bending stress, safety factor) and on Beam3D (CrossSection geometry,
direction-aware loads in y/z/x, torques, diagrams, twist, axial /
torsional / bending stresses, the recompute-on-change guarantee, every
plot method and the pictorial draw()).

Run with: PYTHONPATH=. python examples/beam_analysis.py
"""

import math

from sympy import Symbol

from mecapy.beams import Beam, Beam3D, CrossSection

X = Symbol("x")


def analyze_2d_beam():
    """The 2D Beam: point/moment/distributed/triangular loads, solved
    results, max moment/deflection and bending stress."""
    # 6 m steel beam; E comes from the material database, I is given.
    beam = Beam(length=6, material="steel", second_moment=8.0e-6)

    # Pin at the left end, roller at the right end.
    beam.add_support(0, "pin").add_support(6, "roller")

    # Discrete load types (negative = downward):
    beam.add_point_load(-2000, 3)              # 2 kN at midspan
    beam.add_moment(500, 1)                    # 500 N*m concentrated moment
    beam.add_distributed_load(-300, 0, 2)      # 300 N/m uniform over [0, 2]
    beam.add_triangular_load(-100, 4, 6)       # ramp (order 1) over [4, 6]

    location, moment = beam.max_bending_moment()
    defl_location, deflection = beam.max_deflection()
    stress = beam.bending_stress(distance_to_fiber=0.1)

    print("=" * 60)
    print("2D Beam analysis")
    print("=" * 60)
    print(f"Beam: {beam}")
    print(f"  Elastic modulus (E):  {beam.elastic_modulus / 1e9:.0f} GPa")
    print()
    print("  Reactions:")
    for symbol, value in beam.reactions.items():
        print(f"    {symbol} = {float(value):.1f}")
    print(f"  Shear at x=3-:        {float(beam.shear_force().subs(X, 2.99)):.1f} N")
    print(f"  Moment at x=3:        {float(beam.bending_moment().subs(X, 3)):.1f} N*m")
    print(f"  Slope at x=0:         {float(beam.slope().subs(X, 0)):.6f} rad")
    print(f"  Max moment:           {float(moment):.1f} N*m at x = {float(location):.2f} m")
    print(f"  Max deflection:       {float(deflection) * 1e3:.2f} mm "
          f"at x = {float(defl_location):.2f} m")
    print(f"  Max bending stress:   {float(stress) / 1e6:.2f} MPa")
    print(f"  Safety factor:        {beam.safety_factor(abs(float(stress))):.2f}")


def function_load_2d_beam():
    """add_function_load on the 2D Beam: an arbitrary q = f(x) in N/m,
    discretized into constant segments. With many segments the symbolic
    max_* searches get slow, so evaluate the diagrams pointwise."""
    beam = Beam(length=6, material="steel", second_moment=8.0e-6)
    beam.add_support(0, "pin").add_support(6, "roller")
    # Half-sine load, 150 N/m peak — total = 2 * 150 * 6 / pi ≈ 573 N.
    beam.add_function_load(lambda x: -150 * math.sin(math.pi * x / 6), segments=30)

    print("\n" + "=" * 60)
    print("2D Beam with a function load q = -150*sin(pi*x/6)")
    print("=" * 60)
    total = 2 * 150 * 6 / math.pi
    print(f"  Total load (analytic):  {total:.0f} N")
    print("  Reactions:")
    for symbol, value in beam.reactions.items():
        print(f"    {symbol} = {float(value):.1f}")
    moment = beam.bending_moment()
    deflection = beam.deflection()
    print(f"  Moment at midspan:      {float(moment.subs(X, 3)):.1f} N*m")
    print(f"  Deflection at midspan:  {float(deflection.subs(X, 3)) * 1e3:.2f} mm")


def build_3d_beam():
    """Beam3D: cross-sections and every direction-aware load type."""
    # Solid circular 40 mm section; rectangular shown for comparison.
    circular = CrossSection.circular(0.040)
    rectangular = CrossSection.rectangular(width=0.03, height=0.05)

    print("\n" + "=" * 60)
    print("Beam3D model")
    print("=" * 60)
    print(f"Circular section:    {circular}")
    print(f"Rectangular section: {rectangular}")

    beam = Beam3D(length=2.0, section=circular, material="steel", name="Demo Beam3D")
    print(f"Beam: {beam}")
    print(f"  E = {beam.elastic_modulus / 1e9:.0f} GPa, "
          f"G = {beam.shear_modulus / 1e9:.0f} GPa (from the material database)")

    # Supports restrain both transverse planes (like bearings).
    beam.add_support(0, "pin").add_support(2.0, "roller")

    # Direction-aware loads: y and z bending planes, x = axial.
    beam.add_point_load(-1000, 1.0, dir="y")        # 1 kN down at midspan
    beam.add_point_load(-400, 1.5, dir="z")         # out-of-plane point load
    beam.add_distributed_load(-200, 0.2, 0.8, dir="y")
    beam.add_function_load(                          # q = f(x) in the z plane
        lambda x: -250 * x, start=0, end=2.0, segments=30, dir="z"
    )
    beam.add_point_load(500, 2.0, dir="x")          # axial pull at the tip
    beam.add_moment(150, 0.5, dir="z")              # moment about z (bends plane y)
    beam.add_torque(80, 2.0)                        # torque, grounded at x=0
    return beam


def analyze_3d_results(beam):
    """Beam3D solved results: reactions, diagrams, twist and stresses."""
    section = beam.section
    reactions = beam.reactions
    max_moment = beam.max_bending_moment()
    max_deflection = beam.max_deflection()

    print("\n" + "=" * 60)
    print("Beam3D results")
    print("=" * 60)
    for plane in ("y", "z"):
        total = sum(float(v) for v in reactions[plane].values())
        loc, moment = max_moment[plane]
        dloc, deflection = max_deflection[plane]
        print(f"  Plane {plane}:")
        print(f"    Reactions total:   {total:.1f} N")
        print(f"    Max moment:        {float(moment):.1f} N*m at x = {float(loc):.2f} m")
        print(f"    Max deflection:    {abs(float(deflection)) * 1e3:.3f} mm "
              f"at x = {float(dloc):.2f} m")

    print(f"  Shear V_y at x=0.75:   {float(beam.shear_force()['y'].subs(X, 0.75)):.1f} N")
    print(f"  Axial force at x=1:    {float(beam.axial_force().subs(X, 1.0)):.1f} N")
    print(f"  Torque at x=1:         {float(beam.torsional_moment().subs(X, 1.0)):.1f} N*m")
    print(f"  Twist at tip:          {float(beam.twist().subs(X, 2.0)) * 1e3:.3f} mrad")

    axial = float(beam.axial_stress().subs(X, 1.0))
    torsion = float(beam.torsional_stress().subs(X, 1.0))
    bending = beam.bending_stress("both")
    print(f"  Axial stress:          {axial / 1e6:.3f} MPa")
    print(f"  Torsional stress:      {torsion / 1e6:.2f} MPa")
    for plane in ("y", "z"):
        value = float(bending[plane].subs(X, 1.0))
        print(f"  Bending stress ({plane}):    {value / 1e6:.2f} MPa at midspan")
    print(f"  Safety factor (bend y): "
          f"{beam.safety_factor(abs(float(bending['y'].subs(X, 1.0)))):.2f}")

    # Recompute, don't cache: swapping the section updates every result.
    deflection_before = float(beam.deflection()["y"].subs(X, 1.0))
    beam.section = CrossSection.circular(0.050)
    deflection_after = float(beam.deflection()["y"].subs(X, 1.0))
    beam.section = section  # restore
    print(f"  Midspan deflection d=40 mm -> 50 mm: "
          f"{abs(deflection_before) * 1e3:.3f} -> {abs(deflection_after) * 1e3:.3f} mm "
          "(recomputed automatically)")


def plot_everything(beam):
    """Every Beam3D plot: individual diagrams, stresses, the combined
    stacked figure and the pictorial schematic."""
    beam.draw(show=True)                     # pictorial schematic
    beam.plot_shear_force(show=True)         # dir="both" by default
    beam.plot_bending_moment(dir="y", show=True)
    beam.plot_deflection(show=True)
    beam.plot_axial_stress(show=True)
    beam.plot_torsional_stress(show=True)
    beam.plot_bending_stress(show=True)
    beam.plot_results(show=True)             # stacked 5-panel summary


if __name__ == "__main__":
    analyze_2d_beam()
    function_load_2d_beam()
    beam3d = build_3d_beam()
    analyze_3d_results(beam3d)
    plot_everything(beam3d)
##Cosas que arreglar
##Graficos de viga 3d bastante bueno, 
# pero hay que revisar los graficos de momento todo eso, hay algunos que se pueden unir unos con otros, revisar a mano mejor