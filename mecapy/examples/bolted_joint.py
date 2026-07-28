"""Bolted joint analysis example.

Analyzes a 4-bolt rectangular pattern under combined shear, axial,
torsion and bending loads: first the force distribution over the bolt
group, then the preloaded-joint chain (bolt/member stiffness, joint
constant, resultant bolt load and member load) and the member-side
checks, then plots both.

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
        plates=[(25.0, "steel"), (25.0, "steel")],   # 50 mm grip, M12 x 60 spans it
    )
    print("\n" + "=" * 60)
    print("Bolted union: 4-bolt rectangular pattern (120 x 80 mm)")
    print(f"Forces  (Fx, Fy, Fz) = {union.forces} N")
    print(f"Moments (Mx, My, Mz) = {union.moments} N*mm")
    print(f"Centroid: {union.centroid} mm")
    # Bending resolves about the farthest bolt on the compression side
    # (the plate cannot pull), so every bolt takes tension and the bolt
    # on the pivot line takes none of that moment.
    pivots = union.bending_pivots
    print(f"Bending reference: {union.bending_reference} "
          f"(pivot Mx: #{pivots['x']}, pivot My: #{pivots['y']})")

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

    # ---- Preloaded joint: how the external load splits ----
    # The bolt and the clamped members are springs in parallel, so an
    # external tension P per bolt only adds C*P to the bolt (the rest
    # decompresses the members). C is small here, which is exactly why a
    # preloaded joint is fatigue-friendly.
    print("\n" + "=" * 60)
    print("Preloaded joint (two 25 mm steel plates, 50 mm grip)")
    print(f"Grip (l):              {union.grip:.1f} mm")
    print(f"Bolt stiffness (kb):   {union.bolt_stiffness:.0f} N/mm  (Shigley eq. 8-17)")
    print(f"Member stiffness (km): {union.member_stiffness:.0f} N/mm (30 deg frusta)")
    print(f"Joint constant (C):    {union.joint_constant:.4f}")
    print(f"Preload (Fi):          {union.effective_preload / 1000:.1f} kN")

    print(f"\n{'Bolt':>4} {'P [N]':>10} {'Fb [N]':>10} {'Fm [N]':>11} "
          f"{'n_sep':>7} {'n_proof':>8} {'n_slip':>7}")
    separation = union.separation_safety_factors()
    proof = union.proof_safety_factors()
    slip = union.slip_safety_factors()
    for number, entry in union.bolt_tensions().items():
        print(f"{number:>4} {entry['external']:>10.1f} "
              f"{entry['bolt_tension']:>10.1f} {entry['member_force']:>11.1f} "
              f"{separation[number]:>7.2f} {proof[number]:>8.2f} "
              f"{slip[number]:>7.2f}")

    # ---- Member-side checks ----
    # The members fail differently from the bolt: the hole crushes, the
    # clamp runs out, or the bolt tears out to a free edge. The last one
    # is reported as the edge distance you must provide, not as an input.
    print(f"\n{'Bolt':>4} {'clamp [N]':>11} {'n_bear':>8} {'n_washer':>9} "
          f"{'e_min [mm]':>11}")
    bearing = union.bearing_safety_factors()
    edges = union.minimum_edge_distances(safety_factor=2.0)
    for number, state in union.clamp_states().items():
        print(f"{number:>4} {state['clamp']:>11.1f} {bearing[number]:>8.2f} "
              f"{state['washer_safety_factor']:>9.2f} "
              f"{edges[number]['edge_distance']:>11.2f}")
    print("(e_min for a tear-out safety factor of 2, from the bolt centre)")

    print("\n" + union.describe())

    # ---- Plots ----
    try:
        union.plot_distribution()
        union.plot_tension()
    except ImportError:
        print("\n(matplotlib not installed - skipping the plots)")


if __name__ == "__main__":
    main()
