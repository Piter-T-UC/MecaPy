"""Example: AGMA gear rating, planetary set and worm drive with MecaPy."""

from mecapy.gears import (
    HelicalGear,
    PlanetaryGearSet,
    SpurGear,
    Worm,
    WormWheel,
)


def rate_helical_pair():
    """AGMA bending and pitting rating of a helical mesh."""
    pinion = HelicalGear(teeth=20, module=3.0, helix_angle=20.0,
                         hand="right", face_width=40.0)
    gear = HelicalGear(teeth=60, module=3.0, helix_angle=20.0,
                       hand="left", face_width=40.0)

    rating = pinion.rate_agma(gear, power_kw=10.0, pinion_speed_rpm=1200,
                              Qv=8, hardness_HB=300, Ko=1.25)

    print(rating.summary())
    print()


def planetary_reducer():
    """Kinematics of a 3-planet epicyclic reducer."""
    ps = PlanetaryGearSet(sun=SpurGear(24, module=2.0),
                          planet=SpurGear(18, module=2.0),
                          ring=SpurGear(60, module=2.0),
                          n_planets=3)

    print("Planetary Gear Set")
    print("=" * 40)
    print(f"{ps!r}")
    print(f"Ratio (ring fixed, sun -> carrier): "
          f"{ps.ratio('sun', 'carrier', 'ring'):.2f}")
    print(f"Ratio (carrier fixed, sun -> ring): "
          f"{ps.ratio('sun', 'ring', 'carrier'):.2f}")
    speeds = ps.speeds("sun", 3500.0, "ring")
    print(f"Speeds for sun at 3500 rpm, ring fixed:")
    for member, rpm in speeds.items():
        print(f"  {member}: {rpm:.1f} rpm")
    print()


def worm_drive():
    """Ratio, efficiency and self-locking check of a worm drive."""
    worm = Worm(starts=2, module=4.0, pitch_diameter=50.0)
    wheel = WormWheel(teeth=40, module=4.0, face_width=30.0)

    print("Worm Drive")
    print("=" * 40)
    print(f"Ratio: {worm.ratio_with(wheel):.0f}:1")
    print(f"Lead angle: {worm.lead_angle:.2f} deg")
    print(f"Center distance: {worm.center_distance_with(wheel):.1f} mm")
    print(f"Efficiency (f=0.05): {worm.efficiency():.1%}")
    print(f"Self-locking (f=0.05): {worm.is_self_locking()}")
    print(f"Permissible wear load: {worm.permissible_load(wheel):.0f} N")
    print()


if __name__ == "__main__":
    rate_helical_pair()
    planetary_reducer()
    worm_drive()
