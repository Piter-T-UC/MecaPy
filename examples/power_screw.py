"""Example: power screw (lead screw) analysis with MecaPy.

Run with: PYTHONPATH=. python examples/power_screw.py
"""

from mecapy.shafts import PowerScrew


def raise_and_lower():
    """Torque, efficiency and self-locking (Shigley Example 8-1)."""
    screw = PowerScrew(major_diameter=32.0, pitch=4.0, length=200.0,
                       n_starts=2, thread_form="square")
    load = 6400.0        # N
    mu = mu_collar = 0.08
    dc = 40.0            # mm, mean collar diameter

    print("=" * 60)
    print("Power screw - raising and lowering a load")
    print("=" * 60)
    print(f"Screw: {screw}")
    print(f"  Lead:           {screw.lead:.1f} mm")
    print(f"  Mean diameter:  {screw.mean_diameter:.1f} mm")
    print(f"  Root diameter:  {screw.minor_diameter:.1f} mm")
    print(f"  Lead angle:     {screw.lead_angle:.2f} deg")

    t_r = screw.raise_torque(load, mu, collar_diameter=dc, mu_collar=mu_collar)
    t_l = screw.lower_torque(load, mu, collar_diameter=dc, mu_collar=mu_collar)
    eff = screw.efficiency(load, mu, collar_diameter=dc, mu_collar=mu_collar)
    print(f"\n  Load F = {load:.0f} N, mu = {mu}, collar dc = {dc} mm")
    print(f"  Torque to raise:  {t_r / 1000:.2f} N*m")
    print(f"  Torque to lower:  {t_l / 1000:.2f} N*m")
    print(f"  Efficiency:       {eff * 100:.1f} %")
    print(f"  Self-locking:     {screw.is_self_locking(mu)}")


def stress_and_buckling():
    """Body, thread-engagement and buckling analysis of a longer screw."""
    screw = PowerScrew(major_diameter=40.0, pitch=6.0, length=600.0,
                       thread_form="acme", material="steel")
    load = 25000.0            # N compressive
    engagement = 48.0        # mm nut height (8 engaged threads)

    print("\n" + "=" * 60)
    print("Power screw - stress and buckling")
    print("=" * 60)
    print(f"Screw: {screw}")

    t_r = screw.raise_torque(load)
    print(f"\n  Load F = {load:.0f} N, engagement H = {engagement:.0f} mm")
    print(f"  Axial stress:        {screw.axial_stress(load):.1f} MPa")
    print(f"  Torsional stress:    {screw.torsional_stress(t_r):.1f} MPa")
    print(f"  Von Mises stress:    {screw.von_mises_stress(load, t_r):.1f} MPa")
    print(f"  Body safety factor:  {screw.screw_safety_factor(load, t_r):.2f}")
    print(f"  Bearing pressure:    {screw.bearing_pressure(load, engagement):.1f} MPa")
    print(f"  Thread bending:      "
          f"{screw.thread_bending_stress(load, engagement):.1f} MPa")
    print(f"  Thread shear:        "
          f"{screw.thread_shear_stress(load, engagement):.1f} MPa")

    p_cr = screw.critical_buckling_load(end_condition=1.0)
    print(f"\n  Slenderness ratio:   {screw.slenderness_ratio():.0f} "
          f"(Euler/Johnson transition {screw.johnson_transition_ratio():.0f})")
    print(f"  Euler critical load: {p_cr / 1000:.1f} kN")
    print(f"  Buckling safety:     {screw.check_buckling(load):.2f}")


if __name__ == "__main__":
    raise_and_lower()
    stress_and_buckling()
