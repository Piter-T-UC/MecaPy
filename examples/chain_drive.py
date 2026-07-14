"""Example: roller chain drive analysis using MecaPy."""

from mecapy.chains import Chain


def analyze_chain_drive():
    """Analyze a roller chain drive (ANSI 40, pitch 12.7 mm)."""
    chain = Chain(pitch=12.7, teeth=17, strands=1)

    driven_teeth = 40
    center_pitches = 30
    speed = 1200.0  # rev/min

    print("Roller Chain Drive Analysis")
    print("=" * 40)
    print(f"Pitch: {chain.pitch} mm, driving teeth: {chain.teeth}")
    print(f"Driver sprocket pitch diameter: {chain.pitch_diameter():.2f} mm")
    print(f"Driven sprocket pitch diameter: {chain.pitch_diameter(driven_teeth):.2f} mm")
    print(f"Chain length: {chain.length_in_pitches(driven_teeth, center_pitches):.1f} pitches")
    print(f"Chain velocity: {chain.velocity(speed) / 1000:.2f} m/s")
    print(f"Chordal speed variation: {chain.chordal_speed_variation() * 100:.2f} %")
    print()


if __name__ == "__main__":
    analyze_chain_drive()
