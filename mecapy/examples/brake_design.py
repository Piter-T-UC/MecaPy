"""Example: clutch/brake sizing with energy and temperature rise.

Run with: PYTHONPATH=. python examples/brake_design.py
"""

import math

from mecapy.brakes import BandBrake, InternalShoeBrake
from mecapy.clutches import DiscClutch, get_friction_material
from mecapy.utils import thermal
from mecapy.wheels import Flywheel


def drum_brake_shigley():
    """Two-shoe internal drum brake (Shigley Example 16-2)."""
    a = math.sqrt(112 ** 2 + 50 ** 2)  # hinge-pin distance from Fig. 16-8
    primary = InternalShoeBrake(drum_radius=150, face_width=32,
                                pivot_distance=a, theta1=0, theta2=126,
                                actuation_arm=212, mu=0.32,
                                rotation="self_energizing")
    secondary = InternalShoeBrake(drum_radius=150, face_width=32,
                                  pivot_distance=a, theta1=0, theta2=126,
                                  actuation_arm=212, mu=0.32,
                                  rotation="de_energizing")
    p_max = 1.0  # MPa lining limit

    print("=" * 60)
    print("Internal drum brake - two shoes (Shigley Ex. 16-2)")
    print("=" * 60)
    force = primary.actuating_force(p_max)
    print(f"  Actuating force:      {force:.0f} N")
    print(f"  Primary shoe torque:  {primary.torque(p_max) / 1e3:.0f} N*m")
    print(f"  Secondary pressure:   "
          f"{secondary.max_pressure_for_force(force):.3f} MPa")
    print(f"  Secondary torque:     "
          f"{secondary.torque_for_force(force) / 1e3:.0f} N*m")
    total = primary.torque(p_max) + secondary.torque_for_force(force)
    print(f"  Total capacity:       {total / 1e3:.0f} N*m")
    print(f"  Self-locking margin:  {primary.self_locking_margin:.2f}")


def disc_clutch_with_thermal():
    """Disc clutch: torque, lining check, engagement heat and cooling."""
    clutch = DiscClutch(outer_diameter=250, inner_diameter=150,
                        n_faces=2, lining="molded")
    lining = get_friction_material("molded")
    force = clutch.actuating_force_uniform_wear(lining["p_max"] / 2)

    print("\n" + "=" * 60)
    print("Disc clutch - capacity and engagement temperature rise")
    print("=" * 60)
    print(f"  Lining: molded (mu = {clutch.mu}, "
          f"p_max = {lining['p_max']} MPa, t_max = {lining['t_max']} C)")
    print(f"  Actuating force:      {force / 1e3:.2f} kN")
    t_wear = clutch.torque_uniform_wear(force)
    t_press = clutch.torque_uniform_pressure(force)
    print(f"  Torque (unif. wear):  {t_wear / 1e3:.0f} N*m")
    print(f"  Torque (unif. press): {t_press / 1e3:.0f} N*m")
    print(f"  Pressure safety:      "
          f"{clutch.pressure_safety_factor(force):.2f}")

    # Engagement: motor side I1 at 1800 rpm couples a stationary load I2.
    inertia_1, inertia_2 = 0.8, 2.4       # kg*m^2
    omega_1 = 1800 * 2 * math.pi / 60     # rad/s
    energy = thermal.clutch_slip_energy(inertia_1, inertia_2, omega_1, 0.0)
    time = thermal.engagement_time(inertia_1, inertia_2, omega_1, 0.0,
                                   t_wear / 1e3)
    delta_t = thermal.temperature_rise(energy, mass=6.0, material="steel")
    print(f"\n  Slip energy:          {energy / 1e3:.2f} kJ")
    print(f"  Engagement time:      {time:.3f} s")
    print(f"  Temperature rise:     {delta_t:.1f} C (6 kg steel assembly)")

    tau = thermal.cooling_time_constant(mass=6.0, specific_heat=500,
                                        h_overall=40.0, area=0.15)
    temp_after = thermal.newton_cooling_temperature(60.0, 20.0 + delta_t,
                                                    20.0, tau)
    print(f"  Cooling time const.:  {tau:.0f} s")
    print(f"  Temp. 1 min later:    {temp_after:.1f} C (ambient 20 C)")


def band_brake_and_flywheel():
    """Band brake stopping a flywheel: tension, stresses, stop heat."""
    band = BandBrake(drum_diameter=400, band_width=80, wrap_angle=270,
                     lining="woven", band_thickness=3.0)
    flywheel = Flywheel(outer_radius=0.4, inner_radius=0.1, thickness=0.08)
    omega = 900 * 2 * math.pi / 60  # rad/s

    print("\n" + "=" * 60)
    print("Band brake stopping a flywheel")
    print("=" * 60)
    print(f"  Flywheel mass:        {flywheel.mass:.0f} kg")
    print(f"  Moment of inertia:    {flywheel.moment_of_inertia:.2f} kg*m^2")
    print(f"  Kinetic energy:       "
          f"{flywheel.kinetic_energy(omega) / 1e3:.1f} kJ at 900 rpm")
    print(f"  Burst safety factor:  {flywheel.burst_safety_factor(omega):.0f}")
    print(f"  Max speed:            {flywheel.max_speed_rpm():.0f} rpm")

    p1 = band.tight_tension_for_pressure(get_friction_material("woven")["p_max"])
    print(f"\n  Tension ratio:        {band.tension_ratio:.2f}")
    print(f"  Tight tension:        {p1 / 1e3:.2f} kN at lining p_max")
    print(f"  Braking torque:       {band.torque(p1) / 1e3:.0f} N*m")
    print(f"  Band stress:          {band.band_stress(p1):.0f} MPa "
          f"(safety {band.band_safety_factor(p1):.1f})")

    energy = thermal.stop_energy(flywheel.moment_of_inertia, omega)
    delta_t = thermal.temperature_rise(energy, mass=15.0, material="steel")
    print(f"  Stop energy:          {energy / 1e3:.1f} kJ")
    print(f"  Drum temp. rise:      {delta_t:.1f} C (15 kg steel drum)")


if __name__ == "__main__":
    drum_brake_shigley()
    disc_clutch_with_thermal()
    band_brake_and_flywheel()
