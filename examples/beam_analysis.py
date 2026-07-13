"""Example: simply supported beam analysis using MecaPy's SymPy-backed Beam."""

from mecapy.beams import Beam


def analyze_simply_supported_beam():
    """Analyze a simply supported steel beam with a central point load."""
    # 6 m steel beam; E is taken from the material database, I is given.
    beam = Beam(length=6.0, material="steel", second_moment=8.0e-6)

    # Pin at the left end, roller at the right end.
    beam.add_support(0, "pin").add_support(6, "roller")

    # 2 kN downward point load at midspan.
    beam.add_point_load(-2000, 3)

    location, moment = beam.max_bending_moment()
    stress = beam.bending_stress(distance_to_fiber=0.1)

    print("Simply Supported Beam Analysis")
    print("=" * 40)
    print(f"Length: {beam.length} m")
    print(f"Material: {beam.material}")
    print(f"Elastic modulus (E): {beam.elastic_modulus / 1e9:.0f} GPa")
    print()
    print(f"Reactions: {beam.reactions}")
    print(f"Max bending moment: {moment} N*m at x = {location} m")
    print(f"Max bending stress: {float(stress) / 1e6:.2f} MPa")
    print(f"Safety factor: {beam.safety_factor(float(stress)):.2f}")
    print()


if __name__ == "__main__":
    analyze_simply_supported_beam()
