"""Internal (ring) gears: geometry, kinematics and a scaled drawing.

Run with ``PYTHONPATH=. python examples/internal_gear_train.py`` from the
project directory (or after ``pip install -e ".[viz]"``).
"""

import matplotlib.pyplot as plt

from mecapy.gears import PlanetaryGearSet, SpurGear, Transmission


def ring_geometry():
    """Compare an internal gear with the external gear of the same size."""
    ring = SpurGear(80, module=2.0, internal=True, name="ring")
    external = SpurGear(80, module=2.0, name="external")

    print("Internal vs external, 80 teeth, module 2 mm")
    print("-" * 46)
    for gear in (external, ring):
        print(f"{gear.name:>9}: d = {gear.pitch_diameter:7.2f} mm, "
              f"da = {gear.outside_diameter:7.2f} mm, "
              f"df = {gear.root_diameter:7.2f} mm")
    print("The ring's tip circle is inside its pitch circle, and its root")
    print("circle outside it - the teeth point inwards.\n")

    pinion = SpurGear(20, module=2.0)
    print(f"Center distance 20-in-80 : "
          f"{pinion.center_distance_with(ring):.2f} mm  (a difference)")
    print(f"Center distance 20-and-80: "
          f"{pinion.center_distance_with(external):.2f} mm  (a sum)")
    print(f"Contact ratio, internal  : "
          f"{pinion.contact_ratio_with(ring):.3f}")
    print(f"Contact ratio, external  : "
          f"{pinion.contact_ratio_with(external):.3f}")
    print(f"Trimming interference    : "
          f"{pinion.has_trimming_interference_with(ring)}\n")


def ring_drive():
    """A single internal stage: same rotation direction in and out."""
    pinion = SpurGear(20, module=2.0, face_width=25.0,
                      speed_rpm=1800.0, power_kw=5.0)
    ring = SpurGear(80, module=2.0, face_width=25.0, internal=True)
    train = Transmission(name="Ring drive").add_stage(pinion, ring)

    print("Ring drive, 20-tooth pinion inside an 80-tooth ring")
    print("-" * 46)
    print(f"overall ratio : {train.overall_ratio:.3f}")
    print(f"train value   : {train.train_value:+.4f}  "
          f"(positive: output turns with the input)")
    print(f"output speed  : {train.output_speed(1800.0):.1f} rpm")
    for stage in train.rotation_senses():
        print(f"rotation      : driver {stage['driver']:+d}, "
              f"driven {stage['driven']:+d}")

    rating = train.rate_agma(hardness_HB=250)[0]
    print(f"AGMA (approximate for internal meshes): "
          f"SF = {rating.SF_pinion:.2f}, SH = {rating.SH:.2f}\n")
    return train


def compound_train():
    """An external stage followed by an internal one."""
    a = SpurGear(17, module=2.0, face_width=25.0,
                 speed_rpm=1500.0, power_kw=7.0)
    b = SpurGear(51, module=2.0, face_width=25.0)
    c = SpurGear(18, module=2.5, face_width=30.0)
    d = SpurGear(72, module=2.5, face_width=30.0, internal=True)
    train = Transmission(name="Compound").add_stage(a, b).add_stage(c, d)

    print("Compound train: 17/51 external, then 18-in-72 internal")
    print("-" * 46)
    print(f"overall ratio : {train.overall_ratio:.3f}")
    print(f"train value   : {train.train_value:+.4f}")
    for item in train.stage_layout():
        kind = " internal" if item["element"].internal else ""
        print(f"  {item['role']:>6} {item['element'].teeth:3d}t{kind:9s} "
              f"at x = {item['center'][0]:7.2f} mm, "
              f"sense {item['sense']:+d}, "
              f"{item['speed_rpm']:.0f} rpm")
    print()
    return train


def planetary_ring():
    """The ring of a planetary set is a real internal gear."""
    sun = SpurGear(24, module=2.0)
    planet = SpurGear(18, module=2.0)
    gearset = PlanetaryGearSet(sun, planet, 60, n_planets=3,
                               name="epicyclic")

    print("Planetary set 24 / 18 / 60, 3 planets")
    print("-" * 46)
    print(f"ring is internal : {gearset.ring.internal}")
    print(f"ring pitch dia   : {gearset.ring.pitch_diameter:.2f} mm")
    print(f"ring tip dia     : {gearset.ring.outside_diameter:.2f} mm")
    print(f"carrier radius   : {gearset.carrier_radius:.2f} mm")
    print(f"sun -> carrier   : {gearset.ratio('sun', 'carrier', 'ring'):.3f}")
    print()


def save_plot(ring, compound):
    """Draw both trains to scale and save the figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ring.plot(show=False, ax=axes[0])
    compound.plot(show=False, ax=axes[1])
    fig.tight_layout()
    fig.savefig("internal_gear_train.png", dpi=110)
    print("Saved internal_gear_train.png")


if __name__ == "__main__":
    ring_geometry()
    ring = ring_drive()
    compound = compound_train()
    planetary_ring()
    save_plot(ring, compound)
