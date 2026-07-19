"""Example: rolling-contact and journal bearing design using MecaPy.

Walks through the Shigley ch. 11 rating-life procedure (life, reliability,
combined loading, duty cycles) and the ch. 12 hydrodynamic design procedure
(viscosity, Petroff, Raimondi-Boyd charts, temperature iteration), and
renders the classic textbook figures to bearing_design_charts.png.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from mecapy.bearings import Bearing, JournalBearing
from mecapy.bearings.bearing_data import weibull_reliability
from mecapy.bearings.lubrication_data import RAIMONDI_BOYD, raimondi_boyd, viscosity

# Fixed categorical hue order (colorblind-safe, never cycled).
COLORS = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#a463f2"]
GRID = dict(color="#d5d5d5", linewidth=0.6)


def rolling_bearing_design():
    """Size and analyse a deep-groove ball bearing (Shigley ch. 11)."""
    bearing = Bearing(
        bore_diameter=25.0,
        outer_diameter=52.0,
        width=15.0,
        bearing_type="ball",
        C10=35000.0,
        C0=14000.0,
        name="6205",
    )

    radial = 3000.0  # N
    axial = 1400.0  # N
    speed = 1500.0  # rpm

    print("Rolling-Contact Bearing Analysis (Shigley ch. 11)")
    print("=" * 52)
    print(f"Bearing: {bearing.name}  {bearing!r}")
    print(
        f"Ratings: C10 = {bearing.C10 / 1000:.1f} kN, "
        f"C0 = {bearing.C0 / 1000:.1f} kN"
    )
    print()

    equivalent = bearing.equivalent_load(radial, axial)
    print(f"Combined load Fr = {radial:.0f} N, Fa = {axial:.0f} N")
    print(f"  Equivalent radial load P = {equivalent:.0f} N (table 11-1)")

    l10 = bearing.life(equivalent)
    print(
        f"  L10 life: {l10 / 1e6:.1f} Mrev = "
        f"{bearing.life_hours(equivalent, speed):.0f} h at {speed:.0f} rpm"
    )

    l99 = bearing.adjusted_life(equivalent, reliability=0.99)
    print(
        f"  Life at 99% reliability: {l99 / 1e6:.1f} Mrev "
        f"(Weibull adjustment, eq. 11-6)"
    )

    target_hours = 10000.0
    target_rev = target_hours * 60.0 * speed
    c10_needed = bearing.required_C10(
        equivalent, target_rev, reliability=0.99, application_factor=1.2
    )
    print(
        f"  C10 needed for {target_hours:.0f} h at 99% reliability "
        f"with af = 1.2: {c10_needed / 1000:.1f} kN "
        f"({'OK' if c10_needed <= bearing.C10 else 'undersized'})"
    )

    duty = [(2000.0, 0.4, 1500.0), (4000.0, 0.4, 900.0), (6000.0, 0.2, 300.0)]
    feq = bearing.equivalent_steady_load(duty)
    print("  Duty cycle (load N, time fraction, rpm):")
    for load, fraction, rpm in duty:
        print(f"    {load:6.0f} N  {fraction:4.0%}  {rpm:5.0f} rpm")
    print(f"  Cubic-mean equivalent load: {feq:.0f} N (eq. 11-15)")
    print()
    return bearing


def journal_bearing_design():
    """Design pass on a hydrodynamic journal bearing (Shigley ch. 12)."""
    r, c, length = 25.0, 0.025, 50.0  # mm
    speed = 25.0  # rev/s
    load = 2500.0  # N
    sae, t_inlet = 40, 60.0  # oil grade, inlet temperature degC

    print("Journal Bearing Design (Shigley ch. 12)")
    print("=" * 52)
    print(
        f"r = {r} mm, c = {c} mm, l = {length} mm, l/d = "
        f"{length / (2 * r):.2f}, N = {speed} rev/s, W = {load} N"
    )
    print(f"Lubricant: SAE {sae}, inlet {t_inlet:.0f} degC")

    # Iterate the mean film temperature T_avg = T_in + dT/2, damped to
    # avoid the oscillation of the plain fixed-point update.
    t_avg = t_inlet
    for iteration in range(20):
        journal = JournalBearing(
            r, c, length, speed, load, sae_grade=sae, temperature=t_avg
        )
        dt = journal.temperature_rise()
        t_avg_new = 0.5 * (t_avg + (t_inlet + dt / 2.0))
        print(
            f"  Iteration {iteration + 1}: T_avg = {t_avg:6.1f} degC, "
            f"mu = {journal.viscosity:5.1f} mPa*s, "
            f"S = {journal.sommerfeld:.3f}, dT = {dt:5.1f} degC"
        )
        converged = abs(t_avg_new - t_avg) < 0.1
        t_avg = t_avg_new
        if converged:
            break

    journal = JournalBearing(
        r, c, length, speed, load, sae_grade=sae, temperature=t_avg
    )
    perf = journal.performance()
    print()
    print(
        f"  Pressure P = {journal.pressure:.2f} MPa, "
        f"Sommerfeld S = {journal.sommerfeld:.3f}"
    )
    print(f"  Petroff friction estimate f = {journal.petroff_friction():.4f}")
    print(
        f"  Minimum film h0 = {perf['h0'] * 1000:.1f} um "
        f"(h0/c = {perf['h0_over_c']:.2f}, "
        f"eccentricity ratio {perf['eccentricity_ratio']:.2f})"
    )
    print(
        f"  Friction coefficient f = {perf['friction_coefficient']:.4f}, "
        f"power loss = {perf['power_loss']:.1f} W"
    )
    print(
        f"  Oil flow Q = {perf['flow'] / 1000:.1f} cm^3/s, "
        f"side flow {perf['side_flow_ratio']:.0%}"
    )
    print(f"  Peak film pressure pmax = {perf['pmax']:.2f} MPa")
    print(f"  Thick-film (fig. 12-4): " f"{'yes' if journal.is_thick_film() else 'no'}")

    trumpler = journal.trumpler_check(startup_load=load, max_temperature=t_avg)
    verdict = {True: "pass", False: "FAIL", None: "skipped"}
    print("  Trumpler criteria (sec. 12-14):")
    for criterion, status in trumpler.items():
        print(f"    {criterion:20s} {verdict[status]}")
    print()
    return journal


def plot_charts():
    """Render the Shigley-style figures to bearing_design_charts.png."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Bearing design charts (Shigley ch. 11 and 12)", fontsize=13)

    # 1. Weibull reliability vs dimensionless life (sec. 11-4).
    ax = axes[0, 0]
    x = np.logspace(-2, 1, 300)
    ax.semilogx(x, [weibull_reliability(v) for v in x], color=COLORS[0], linewidth=2)
    ax.plot([1.0], [weibull_reliability(1.0)], "o", color=COLORS[0], markersize=8)
    ax.annotate(
        "L10: R = 0.90 at x = 1", xy=(1.0, 0.90), xytext=(1.6, 0.97), fontsize=9
    )
    ax.set_xlabel("life x = L / L10")
    ax.set_ylabel("reliability R")
    ax.set_title("Weibull bearing-life reliability")
    ax.grid(True, **GRID)

    # 2. Viscosity-temperature curves for SAE oils (fig. 12-13).
    ax = axes[0, 1]
    temps = np.linspace(20.0, 140.0, 200)
    for color, grade in zip(COLORS, (10, 20, 30, 40, 50, 60)):
        mu = [viscosity(grade, t) for t in temps]
        ax.semilogy(temps, mu, color=color, linewidth=2)
        ax.annotate(
            f"SAE {grade}",
            xy=(temps[8], mu[8]),
            fontsize=8,
            color=color,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("temperature (degC)")
    ax.set_ylabel("absolute viscosity (mPa*s)")
    ax.set_title("Oil viscosity vs temperature")
    ax.grid(True, **GRID)

    # 3 & 4. Raimondi-Boyd charts (figs. 12-16 and 12-18).
    ld_labels = [
        ("inf", "l/d = inf"),
        (1.0, "l/d = 1"),
        (0.5, "l/d = 1/2"),
        (0.25, "l/d = 1/4"),
    ]
    for ax, field, title, ylabel, log_y in (
        (
            axes[1, 0],
            "h0_over_c",
            "Minimum film thickness (fig. 12-16)",
            "h0 / c",
            False,
        ),
        (
            axes[1, 1],
            "friction_variable",
            "Friction variable (fig. 12-18)",
            "(r/c) f",
            True,
        ),
    ):
        for color, (key, label) in zip(COLORS, ld_labels):
            rows = sorted(RAIMONDI_BOYD[key], key=lambda row: row[1])
            s_grid = np.logspace(math.log10(rows[0][1]), math.log10(rows[-1][1]), 120)
            ld = math.inf if key == "inf" else key
            values = [raimondi_boyd(s, ld)[field] for s in s_grid]
            plot = ax.loglog if log_y else ax.semilogx
            plot(s_grid, values, color=color, linewidth=2, label=label)
        ax.set_xlabel("Sommerfeld number S")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", **GRID)
        ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig("bearing_design_charts.png", dpi=150, bbox_inches="tight")
    print("Charts saved to bearing_design_charts.png")


if __name__ == "__main__":
    rolling_bearing_design()
    journal_bearing_design()
    plot_charts()
