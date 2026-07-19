"""Bolted joint analysis example.

Analyzes a 4-bolt rectangular pattern under combined shear, axial,
torsion and bending loads, then plots the force distribution.

Run with: PYTHONPATH=. python examples/bolted_joint.py
"""

from mecapy.bolts import Bolt, BoltedUnion


def main():
    print("=" * 60)
    print("MecaPy - Bolted Joint Example")
    print("=" * 60)

    # ---- Single bolt properties ----
    bolt = Bolt(size="M12", length=60.0, property_class="8.8")
    print(f"\nBolt: {bolt}")
    print(f"  Nominal diameter:    {bolt.nominal_diameter:.1f} mm")
    print(f"  Pitch:               {bolt.pitch:.2f} mm")
    print(f"  Stress area As:      {bolt.stress_area:.1f} mm^2")
    print(f"  Nominal area:        {bolt.nominal_area:.1f} mm^2")
    print(f"  Proof load:          {bolt.proof_load / 1000:.1f} kN")
    print(f"  Recommended preload: {bolt.recommended_preload / 1000:.1f} kN")
    print(f"  Axial stiffness:     {bolt.stiffness / 1000:.0f} kN/mm")
    force = 20000.0
    print(f"\n  Under F = {force / 1000:.0f} kN:")
    print(f"    Tensile stress:  {bolt.tensile_stress(force):.1f} MPa")
    print(f"    Elongation:      {bolt.elongation(force):.4f} mm")
    print(f"    Safety factor:   {bolt.bolt_safety_factor(force):.2f}")

    # ---- Bolt group: 4 bolts on a 120 x 80 mm rectangle ----
    positions = [
        [1, 0.0, 0.0],
        [2, 120.0, 0.0],
        [3, 120.0, 80.0],
        [4, 0.0, 80.0],
    ]
    union = BoltedUnion(
        bolt,
        positions,
        forces=(3000.0, 2000.0, 12000.0),      # N: in-plane shear + axial
        moments=(4e5, -2e5, 8e5),              # N*mm: bending Mx, My + torsion Mz
    )
    print("\n" + "=" * 60)
    print("Bolted union: 4-bolt rectangular pattern (120 x 80 mm)")
    print(f"Forces  (Fx, Fy, Fz) = {union.forces} N")
    print(f"Moments (Mx, My, Mz) = {union.moments} N*mm")
    print(f"Centroid: {union.centroid} mm")

    print(f"\n{'Bolt':>4} {'Fsx [N]':>10} {'Fsy [N]':>10} {'|Fs| [N]':>10} "
          f"{'Axial [N]':>10} {'SF':>7}")
    factors = union.safety_factors()
    for number, entry in union.bolt_forces().items():
        fsx, fsy = entry["shear"]
        print(f"{number:>4} {fsx:>10.1f} {fsy:>10.1f} "
              f"{entry['shear_magnitude']:>10.1f} {entry['axial']:>10.1f} "
              f"{factors[number]:>7.2f}")

    number, entry = union.max_loaded_bolt()
    print(f"\nMost loaded bolt: #{number} "
          f"(|Fs| = {entry['shear_magnitude']:.1f} N, axial = {entry['axial']:.1f} N)")

    # ---- Plot ----
    try:
        union.plot_distribution()
    except ImportError:
        print("\n(matplotlib not installed - skipping the distribution plot)")


if __name__ == "__main__":
    main()
