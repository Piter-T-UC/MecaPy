"""Welded joint analysis example.

Demonstrates single welds, weld groups, and stress analysis under
combined shear, axial, and torsional loading using the elastic
"weld as a line" method (Shigley Ch. 9).

Run with: python examples/welded_joint.py
"""

from mecapy.welds import Weld, WeldedUnion, WeldLine, WeldCircle


def main():
    print("=" * 70)
    print("MecaPy - Welded Joint Example")
    print("=" * 70)

    # ---- Single weld properties ----
    print("\n--- Single Weld Examples ---\n")

    # Example 1: Fillet weld on a straight line
    weld_line = WeldLine(start=(0, 0), end=(100, 0))
    fillet_weld = Weld(
        path=weld_line,
        size=5,              # 5 mm fillet leg
        weld_type="fillet",
        electrode="E70xx"
    )
    print(f"Fillet weld (horizontal line):")
    print(f"  Path: {fillet_weld.path}")
    print(f"  Length: {fillet_weld.length:.1f} mm")
    print(f"  Centroid: {fillet_weld.centroid}")
    print(f"  Weld size: {fillet_weld.size} mm (fillet leg)")
    print(f"  Throat: {fillet_weld.throat:.2f} mm (0.707 × leg)")
    print(f"  Throat area: {fillet_weld.throat_area:.1f} mm²")
    print(f"  Electrode: {fillet_weld.electrode}")
    print(f"  Allowable stress: {fillet_weld.allowable_stress:.1f} MPa")

    # Example 2: Butt weld on a vertical line
    weld_line2 = WeldLine(start=(0, 0), end=(0, 50))
    butt_weld = Weld(
        path=weld_line2,
        size=8,              # 8 mm plate thickness (full penetration)
        weld_type="butt",
        electrode="E70xx"
    )
    print(f"\nButt weld (vertical line):")
    print(f"  Length: {butt_weld.length:.1f} mm")
    print(f"  Weld size: {butt_weld.size} mm (plate thickness)")
    print(f"  Throat: {butt_weld.throat:.2f} mm (full penetration = size)")
    print(f"  Throat area: {butt_weld.throat_area:.1f} mm²")
    print(f"  Allowable stress: {butt_weld.allowable_stress:.1f} MPa")

    # Example 3: Circular weld
    weld_circle = WeldCircle(center=(50, 50), radius=30)
    circular_weld = Weld(
        path=weld_circle,
        size=6,
        weld_type="fillet",
        electrode="E70xx"
    )
    print(f"\nCircular fillet weld:")
    print(f"  Path center: {circular_weld.centroid}")
    print(f"  Radius: {weld_circle.radius} mm")
    print(f"  Length (circumference): {circular_weld.length:.1f} mm")
    print(f"  Throat area: {circular_weld.throat_area:.1f} mm²")

    # ---- Weld group (union) analysis ----
    print("\n" + "=" * 70)
    print("--- Welded Union Under Combined Loading ---\n")

    # Create a T-joint: two vertical welds welding a vertical plate
    # to a horizontal plate
    weld1 = Weld(
        path=WeldLine(start=(20, 0), end=(20, 80)),
        size=5,
        weld_type="fillet",
        electrode="E70xx"
    )
    weld2 = Weld(
        path=WeldLine(start=(80, 0), end=(80, 80)),
        size=5,
        weld_type="fillet",
        electrode="E70xx"
    )

    # Create the welded union with combined loading:
    # - Shear force in x direction
    # - Axial force (out of plane, z direction)
    # - Torsion moment about z axis
    union = WeldedUnion(
        welds=[weld1, weld2],
        forces=(5000.0, 0.0, 8000.0),      # Fx, Fy, Fz in N
        moments=(0.0, 0.0, 15000.0),       # Mx, My, Mz in N*mm (torsion)
        name="T-Joint"
    )

    print(f"Welded union: {union.name}")
    print(f"Number of welds: {union.n_welds}")
    print(f"Total weld length: {union.total_length:.1f} mm")
    print(f"Total throat area: {union.throat_area:.1f} mm²")
    print(f"Group centroid: {union.centroid}")
    print(f"Unit second moments (Iux, Iuy): {union.unit_second_moment}")
    print(f"Unit polar moment: {union.unit_polar_moment:.1f} mm³")
    print(f"\nApplied loads:")
    print(f"  Forces (Fx, Fy, Fz): {union.forces} N")
    print(f"  Moments (Mx, My, Mz): {union.moments} N*mm")

    # ---- Stress analysis ----
    print("\n" + "=" * 70)
    print("--- Stress Analysis ---\n")

    # Compute stresses at sample points along each weld
    weld_results = union.weld_stresses(n=12)  # 12 sample points per weld

    for i, result in enumerate(weld_results):
        print(f"Weld {i + 1}:")
        print(f"  Max von Mises stress: {result['max_stress']:.1f} MPa")
        print(f"  Max stress location: {result['max_point']}")

    # Get the most stressed point in the entire group
    weld_idx, point, stress = union.max_stress(n=12)
    print(f"\nMost stressed weld: #{weld_idx + 1}")
    print(f"  Location: {point}")
    print(f"  Stress: {stress:.1f} MPa")

    # ---- Safety factors ----
    print("\n" + "=" * 70)
    print("--- Safety Factors ---\n")

    safety_factors = union.safety_factors(n=12)
    for weld_idx, sf in safety_factors.items():
        if sf == float("inf"):
            sf_str = "∞"
        else:
            sf_str = f"{sf:.2f}"
        print(f"Weld {weld_idx + 1}: SF = {sf_str}")

    # ---- Resize and recheck ----
    print("\n" + "=" * 70)
    print("--- Resizing the Welds ---\n")

    print("Original size: 5 mm")
    print(f"Original max stress: {union.max_stress(n=12)[2]:.1f} MPa")

    # Increase weld size to reduce stress
    union.set_size(7)
    print(f"\nAfter increasing size to 7 mm:")
    print(f"New max stress: {union.max_stress(n=12)[2]:.1f} MPa")
    print(f"New safety factors: {union.safety_factors(n=12)}")

    # Or solve directly for the size that gives a target safety factor
    required = union.required_size(n=12, safety_factor=2.0)
    print(f"\nRequired size for SF = 2.0: {required:.2f} mm")

    # ---- Complex weld group example ----
    print("\n" + "=" * 70)
    print("--- Complex Weld Group: Box Joint ---\n")

    # Four welds forming a square frame
    box_welds = [
        Weld(WeldLine(start=(0, 0), end=(100, 0)), size=4, electrode="E70xx"),
        Weld(WeldLine(start=(100, 0), end=(100, 100)), size=4, electrode="E70xx"),
        Weld(WeldLine(start=(100, 100), end=(0, 100)), size=4, electrode="E70xx"),
        Weld(WeldLine(start=(0, 100), end=(0, 0)), size=4, electrode="E70xx"),
    ]

    box_union = WeldedUnion(
        welds=box_welds,
        forces=(2000.0, 3000.0, 5000.0),
        moments=(200000.0, 150000.0, 300000.0),
        name="Box Joint"
    )

    print(f"{box_union.name}:")
    print(f"  Number of welds: {box_union.n_welds}")
    print(f"  Total length: {box_union.total_length:.1f} mm")
    print(f"  Centroid: {box_union.centroid}")
    print(f"  Max stress: {box_union.max_stress(n=16)[2]:.1f} MPa")
    print(f"  Min safety factor: {min(box_union.safety_factors(n=16).values()):.2f}")

    # ---- Plot stress distribution ----
    print("\n" + "=" * 70)
    print("--- Plotting Stress Distribution ---\n")

    try:
        import matplotlib.pyplot as plt
        fig = box_union.plot_distribution(n=24, show=False)
        print("Stress distribution plot created (showing box joint example)")
        print("Use plt.show() to display the plot interactively")
        #plt.show()
        plt.close(fig)
    except ImportError:
        print("(matplotlib not installed - skipping stress distribution plot)")


if __name__ == "__main__":
    main()
