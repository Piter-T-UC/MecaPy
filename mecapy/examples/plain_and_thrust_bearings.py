"""Example: boundary-lubricated bushings and hydrodynamic thrust pads.

Two bearing families that the hydrodynamic journal model cannot cover:

* a sleeve bushing running in the boundary/mixed regime, rated on the
  pressure, velocity and PV limits of its liner (Shigley sec. 12-15);
* a tapered-land thrust bearing, solved from the closed-form plane-slider
  Reynolds solution.

Run with ``PYTHONPATH=. python examples/plain_and_thrust_bearings.py``
unless the package is installed.
"""

from mecapy.bearings import (
    BUSHING_MATERIALS,
    PlainBearing,
    ThrustBearing,
    load_coefficient,
)
from mecapy.bearings.thrust import OPTIMUM_TAPER_RATIO


def bushing_design():
    """Rate a boundary-lubricated bushing against its PV limits."""
    bore, length = 25.0, 25.0  # mm
    load, speed = 1000.0, 5.0  # N, rev/s

    print("Boundary-Lubricated Bushing (Shigley sec. 12-15)")
    print("=" * 56)
    bushing = PlainBearing(
        bore_diameter=bore,
        length=length,
        load=load,
        speed=speed,
        bushing_material="cast_bronze",
        name="idler bush",
    )
    print(
        f"d = {bore} mm, l = {length} mm, W = {load:.0f} N, "
        f"N = {speed} rev/s ({bushing.speed_rpm:.0f} rpm)"
    )
    print(
        f"  P = {bushing.pressure:.3f} MPa, V = {bushing.rubbing_velocity:.3f} m/s, "
        f"PV = {bushing.pv:.3f} MPa*m/s"
    )
    print(
        f"  Friction torque {bushing.friction_torque():.0f} N*mm, "
        f"power loss {bushing.power_loss():.1f} W"
    )
    verdict = {True: "pass", False: "FAIL", None: "skipped"}
    print("  Liner limits (cast bronze):")
    for criterion, status in bushing.pv_check(temperature=80.0).items():
        print(f"    {criterion:12s} {verdict[status]}")
    print(
        f"  Margins: pressure {bushing.pressure_safety_factor():.1f}x, "
        f"velocity {bushing.velocity_safety_factor():.1f}x, "
        f"PV {bushing.pv_safety_factor():.1f}x"
    )
    print(
        f"  Headroom: up to {bushing.maximum_load():.0f} N at this speed, "
        f"or {bushing.maximum_speed():.1f} rev/s at this load"
    )

    # PV is usually what binds, so it is what decides the material.
    print()
    print("  Same duty in other liners (PV margin, >1 means acceptable):")
    for material in ("cast_bronze", "filled_ptfe", "nylon", "ptfe"):
        candidate = PlainBearing(bore, length, load, speed, bushing_material=material)
        row = BUSHING_MATERIALS[material]
        print(
            f"    {material:14s} PV_max = {row['pv_max']:.3f} MPa*m/s  "
            f"margin {candidate.pv_safety_factor():6.2f}  "
            f"{'ok' if candidate.pv_check()['pv'] else 'OVER LIMIT'}"
        )
    print()
    return bushing


def thrust_bearing_design():
    """Size a tapered-land thrust bearing from the slider solution."""
    print("Hydrodynamic Thrust Bearing (fixed-incline pads)")
    print("=" * 56)
    bearing = ThrustBearing(
        inner_radius=50.0,
        outer_radius=100.0,
        n_pads=8,
        speed=30.0,
        load=20000.0,
        viscosity=30.0,
        name="collar",
    )
    performance = bearing.performance()
    print(
        f"ri = {bearing.inner_radius:.0f} mm, ro = {bearing.outer_radius:.0f} mm, "
        f"{bearing.n_pads} pads, N = {bearing.speed:.0f} rev/s, "
        f"W = {bearing.load / 1000:.0f} kN"
    )
    print(
        f"  Pad {bearing.pad_width:.1f} x {bearing.pad_length:.1f} mm, "
        f"U = {bearing.sliding_velocity:.1f} m/s, "
        f"mean pressure {bearing.pressure:.2f} MPa"
    )
    print(
        f"  Film: h2 = {performance['film_thickness'] * 1000:.1f} um, "
        f"h1 = {performance['inlet_film'] * 1000:.1f} um, "
        f"pmax = {performance['pmax']:.2f} MPa"
    )
    print(
        f"  Friction f = {performance['friction_coefficient']:.4f}, "
        f"power loss {performance['power_loss']:.0f} W, "
        f"dT = {bearing.temperature_rise():.1f} degC"
    )

    # The taper ratio is a real design variable: load capacity peaks at
    # about 2.2 and falls away on both sides.
    print()
    print("  Taper ratio sweep (load coefficient Kw, film at fixed load):")
    for ratio in (1.2, 1.5, 2.2, 3.0, 5.0):
        candidate = ThrustBearing(
            inner_radius=50.0,
            outer_radius=100.0,
            n_pads=8,
            speed=30.0,
            load=20000.0,
            taper_ratio=ratio,
            viscosity=30.0,
        )
        print(
            f"    a = {ratio:4.1f}  Kw = {load_coefficient(ratio):.4f}  "
            f"h2 = {candidate.film_thickness() * 1000:5.1f} um"
        )
    print(
        f"  Load capacity peaks at a = {bearing.optimum_taper_ratio():.3f} "
        f"(the tabulated {OPTIMUM_TAPER_RATIO:.3f})"
    )
    print()
    print(bearing.describe())
    print()
    return bearing


def main():
    """Run both design passes."""
    bushing_design()
    thrust_bearing_design()


if __name__ == "__main__":
    main()
