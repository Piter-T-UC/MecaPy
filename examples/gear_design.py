"""Example: AGMA spur gear analysis using MecaPy."""

from mecapy.gears import Gear


def analyze_spur_gear():
    """Analyze an AGMA spur gear pinion transmitting 5 kW at 1200 rpm."""
    pinion = Gear(
        teeth=17,
        module=2.5,
        face_width=38,
        pressure_angle=20.0,
        quality_number=6,
        material="steel",
    )

    power = 5000.0   # W
    speed = 1200.0   # rev/min
    gear_ratio = 3.0

    bending = pinion.bending_stress(power, speed, geometry_factor=0.34)
    contact = pinion.contact_stress(power, speed, gear_ratio=gear_ratio)

    print("AGMA Spur Gear Analysis")
    print("=" * 40)
    print(f"Teeth: {pinion.teeth}, module: {pinion.module} mm")
    print(f"Pitch diameter: {pinion.pitch_diameter} mm")
    print(f"Pitch-line velocity: {pinion.pitch_line_velocity(speed):.2f} m/s")
    print(f"Tangential load Wt: {pinion.tangential_load(power, speed):.1f} N")
    print(f"Dynamic factor Kv: {pinion.dynamic_factor(speed):.3f}")
    print()
    print(f"AGMA bending stress: {bending:.1f} MPa")
    print(f"AGMA contact stress: {contact:.1f} MPa")
    print(f"Bending safety factor: {pinion.bending_safety_factor(bending):.2f}")
    print()


if __name__ == "__main__":
    analyze_spur_gear()
