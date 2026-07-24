"""Example: AGMA gear rating, planetary set and worm drive with MecaPy."""

from mecapy.gears import (
    HelicalGear,
    PlanetaryGearSet,
    SpurGear,
    Transmission,
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


def rate_two_stage_train():
    """AGMA rating of every stage of a compound reducer.

    The train propagates its own speeds and powers, so each mesh is
    rated at its own operating point. Stage 1 runs coarser gearing, so
    it gets its own quality number via ``stage_kwargs``.
    """
    pinion = SpurGear(17, module=3.0, face_width=55.0,
                      power_kw=10.0, speed_rpm=1200.0)
    mid_in = SpurGear(51, module=3.0, face_width=55.0)
    mid_out = SpurGear(18, module=4.0, face_width=70.0)
    output = SpurGear(54, module=4.0, face_width=70.0)

    train = (Transmission(name="two-stage reducer")
             .add_stage(pinion, mid_in, efficiency=0.98)
             .add_stage(mid_out, output, efficiency=0.98))

    # Stage 1 is the coarser, slower set, so it gets its own Qv.
    rating_inputs = dict(Qv=8, Ko=1.25, hardness_HB=380,
                         stage_kwargs=[None, {"Qv": 6}])

    print(train.agma_summary(**rating_inputs))
    print()

    governing = train.agma_governing(**rating_inputs)
    print(f"Weakest stage in bending: {governing['SF_stage']} "
          f"(SF = {governing['SF']:.2f})")
    print(f"Weakest stage in pitting: {governing['SH_stage']} "
          f"(SH = {governing['SH']:.2f})")
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
    print("Speeds for sun at 3500 rpm, ring fixed:")
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
    rate_two_stage_train()
    planetary_reducer()
    worm_drive()
