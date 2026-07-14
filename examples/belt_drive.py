"""Example: V-belt drive analysis using MecaPy."""

from mecapy.belts import Belt


def analyze_belt_drive():
    """Analyze a V-belt drive and find its maximum power."""
    belt = Belt(belt_type="v", friction=0.3, mass_per_length=0.4, groove_angle=38)

    large_d, small_d, center = 0.30, 0.15, 0.80  # meters
    velocity = 20.0  # m/s
    max_tension = 600.0  # N

    theta = belt.wrap_angle(large_d, small_d, center)

    print("V-Belt Drive Analysis")
    print("=" * 40)
    print(f"Wrap angle (small pulley): {theta:.3f} rad")
    print(f"Belt length: {belt.belt_length(large_d, small_d, center):.3f} m")
    print(f"Tension ratio T1/T2: {belt.tension_ratio(theta):.2f}")
    print(f"Centrifugal tension @ {velocity} m/s: {belt.centrifugal_tension(velocity):.1f} N")
    print(f"Maximum power: {belt.max_power(max_tension, velocity, theta) / 1000:.2f} kW")
    print()


if __name__ == "__main__":
    analyze_belt_drive()
