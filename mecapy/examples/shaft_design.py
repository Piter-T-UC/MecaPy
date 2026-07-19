"""Example: Shaft and SteppedShaft design with MecaPy.

Covers everything on Shaft (geometry, torsional stress, grooves, stress
profile, plotting) and SteppedShaft (fusing segments with add_shaft,
fillets, stress profile, plotting).

Run with: PYTHONPATH=. python examples/shaft_design.py
"""

from mecapy.shafts import Shaft, SteppedShaft


def plain_shaft():
    """Basic Shaft geometry and torsional stress."""
    shaft = Shaft(diameter=25.0, length=500.0, material="steel")
    torque = 8.0e4  # N*mm

    print("=" * 60)
    print("Plain shaft")
    print("=" * 60)
    print(f"Shaft: {shaft}")
    print(f"  Polar moment J:     {shaft.polar_moment:.1f} mm^4")
    print(f"  Torsional stress:   {shaft.torsional_stress(torque):.1f} MPa")

    # Settable attributes re-validate and every derived value recomputes.
    shaft.diameter = 30.0
    print(f"  After diameter=30:  J = {shaft.polar_moment:.1f} mm^4, "
          f"stress = {shaft.torsional_stress(torque):.1f} MPa")


def shaft_with_grooves():
    """Shaft.add_groove: round (a=0) and flat (a>0) grooves, and the
    resulting stress profile / Kt each groove carries."""
    shaft = Shaft(diameter=30.0, length=200.0, name="Grooved shaft")
    shaft.add_groove(position=60.0, diameter=22.0, radii=3.0)          # round
    shaft.add_groove(position=140.0, diameter=24.0, radii=1.5, a=10.0)  # flat

    print("\n" + "=" * 60)
    print("Shaft with grooves")
    print("=" * 60)
    print(f"Shaft: {shaft}")
    for number, dims, kt_axial, kt_bending, kt_torsion in shaft.grooves:
        position, diameter, radii, width = dims
        print(f"  Groove #{number}: d={diameter} mm at x={position} mm "
              f"(width={width} mm, radii={radii} mm)")
        print(f"    Kt axial={kt_axial:.2f}  bending={kt_bending:.2f}  torsion={kt_torsion:.2f}")

    torque = 1.2e5  # N*mm
    profile = shaft.stress_profile(torque)
    print(f"\n  Torque = {torque / 1e3:.0f} N*m, "
          f"nominal stress range: {min(profile['nominal_stress']):.1f}-"
          f"{max(profile['nominal_stress']):.1f} MPa")
    for peak in profile["peaks"]:
        print(f"  Peak at x={peak['x']} mm: Kt={peak['kt']:.2f}, "
              f"stress={peak['stress']:.1f} MPa")

    return shaft, torque


def stepped_shaft():
    """SteppedShaft: fuse segments with add_shaft (both sides), fillets,
    and grooves inherited from each segment."""
    base = Shaft(diameter=30.0, length=120.0, name="Stepped shaft")
    base.add_groove(position=60.0, diameter=22.0, radii=3.0)  # round groove

    right_end = Shaft(diameter=22.0, length=100.0)

    left_end = Shaft(diameter=35.0, length=80.0)
    left_end.add_groove(position=40.0, diameter=28.0, radii=1.5, a=8.0)  # flat groove

    complete_shaft = (
        SteppedShaft(base)
        .add_shaft(right_end, "fillet", radius=3, side="right")
        .add_shaft(left_end, "fillet", radius=5, side="left")
    )

    print("\n" + "=" * 60)
    print("Stepped shaft (fused with add_shaft)")
    print("=" * 60)
    print(f"Stepped shaft: {complete_shaft}")
    print(f"  Segment diameters (in order): {[s.diameter for s in complete_shaft.segments]} mm")
    print(f"  Boundaries: {complete_shaft.boundaries} mm")
    for number, dims, kt_axial, kt_bending, kt_torsion in complete_shaft.fillets:
        D, d, radius = dims
        print(f"  Fillet #{number}: D={D} mm -> d={d} mm, radius={radius} mm")
        print(f"    Kt axial={kt_axial:.2f}  bending={kt_bending:.2f}  torsion={kt_torsion:.2f}")

    torque = 1.5e5  # N*mm
    profile = complete_shaft.stress_profile(torque)
    print(f"\n  Torque = {torque / 1e3:.0f} N*m")
    for peak in sorted(profile["peaks"], key=lambda p: p["x"]):
        print(f"  {peak['kind']:6s} peak at x={peak['x']:.0f} mm: "
              f"Kt={peak['kt']:.2f}, stress={peak['stress']:.1f} MPa")

    return complete_shaft, torque


def plot_everything(grooved_shaft, grooved_torque, complete_shaft, stepped_torque):
    """Render both a standalone Shaft plot and the fused SteppedShaft plot."""
    fig1 = grooved_shaft.plot(grooved_torque, show=True)
    #fig1.savefig("shaft_design_shaft.png", dpi=150, bbox_inches="tight")
    #print("\n[OK] Saved: shaft_design_shaft.png")

    fig2 = complete_shaft.plot(stepped_torque, show=True)
    #fig2.savefig("shaft_design_stepped_shaft.png", dpi=150, bbox_inches="tight")
    #print("[OK] Saved: shaft_design_stepped_shaft.png")


if __name__ == "__main__":
    plain_shaft()
    grooved_shaft, grooved_torque = shaft_with_grooves()
    complete_shaft, stepped_torque = stepped_shaft()
    plot_everything(grooved_shaft, grooved_torque, complete_shaft, stepped_torque)
