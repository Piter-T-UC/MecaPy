"""Example: Material fatigue analysis (Shigley Ch. 6) using MecaPy.

Covers the whole Material API: database defaults and overrides, the Marin
equation (ka..ke), recompute-on-change design iteration, the S-N (Woehler)
curve, Goodman safety factors, cyclic load cases with Miner's rule, use of
a Material inside any MechaElement, and the two diagrams (saved as PNG).
"""

from mecapy import Material
from mecapy.shafts import Shaft


def basics_and_marin():
    """Build a Material and walk through the Marin endurance limit."""
    # Strengths default from the database (converted to MPa); any of them
    # can be overridden. Here: a ground 690 MPa steel, 25 mm rotating
    # section, 150 degC, 99% reliability, in bending.
    m = Material(
        "steel",
        ultimate_strength=690,        # MPa (override; DB steel is 400)
        surface_finish="ground",      # ka, Table 6-2
        loading="bending",            # kc (and kb rule)
        diameter=25,                  # mm, size factor kb
        temperature=150,              # degC, kd (Table 6-4, interpolated)
        reliability=0.99,             # ke, Table 6-5
        name="shaft steel",
    )

    print("Material & Marin Equation")
    print("=" * 40)
    print(repr(m))
    print(f"Sut = {m.ultimate_strength:.0f} MPa, Sy = {m.yield_strength:.0f} MPa")
    print(f"Se' (unmodified) = {m.se_prime:.1f} MPa")
    print(f"ka (ground)      = {m.ka:.4f}")
    print(f"kb (d=25 mm)     = {m.kb:.4f}")
    print(f"kc (bending)     = {m.kc:.2f}")
    print(f"kd (150 degC)    = {m.kd:.4f}")
    print(f"ke (99%)         = {m.ke:.3f}")
    print(f"Se = ka*kb*kc*kd*ke*Se' = {m.endurance_limit:.1f} MPa")
    print()
    return m


def design_iteration(m):
    """Change inputs after construction — everything recomputes."""
    print("Design Iteration (recompute on change)")
    print("=" * 40)
    print(f"Baseline Se: {m.endurance_limit:.1f} MPa")

    m.surface_finish = "machined"
    print(f"machined instead of ground -> Se = {m.endurance_limit:.1f} MPa")

    m.diameter = 100
    print(f"d = 100 mm instead of 25   -> Se = {m.endurance_limit:.1f} MPa")

    m.temperature = 400
    print(f"400 degC instead of 150    -> Se = {m.endurance_limit:.1f} MPa")

    # Back to the good design
    m.surface_finish = "ground"
    m.diameter = 25
    m.temperature = 150
    print(f"restored                   -> Se = {m.endurance_limit:.1f} MPa")
    print()


def sn_curve_and_goodman(m):
    """S-N (Woehler) curve and Goodman criterion for single stresses."""
    print("S-N Curve & Goodman")
    print("=" * 40)
    print(f"f (fraction at 10^3 cycles) = {m.f:.3f}")
    print(f"S-N fit: S = {m.sn_a:.1f} * N^({m.sn_b:.4f})  [MPa]")

    for sigma_a, sigma_m in [(450, 0), (300, 0), (300, 150), (150, 0)]:
        n = m.cycles_to_failure(sigma_a, sigma_m)
        life = "infinite" if n == float("inf") else f"{n:,.0f} cycles"
        print(f"  sigma_a = {sigma_a} MPa, sigma_m = {sigma_m} MPa -> {life}")

    sf = m.goodman_safety_factor(150, 100)
    print(f"Goodman SF at (sigma_a=150, sigma_m=100): {sf:.2f}")
    print()


def load_cases_and_miner(m):
    """Cyclic load cases: add / edit / remove, Miner's rule."""
    print("Load Cases & Miner's Rule")
    print("=" * 40)

    # Builder-style: add_load_case chains
    m.add_load_case(350, mean_stress=0, cycles=5e3, label="startup") \
     .add_load_case(250, mean_stress=50, cycles=5e5, label="cruise") \
     .add_load_case(120, mean_stress=50, cycles=1e7, label="idle")

    # Edit by label (revalidates), remove by label or index
    m.edit_load_case("cruise", cycles=1e5)
    m.add_load_case(500, cycles=10, label="oops")
    m.remove_load_case("oops")

    for case in m.load_cases:
        n = m.cycles_to_failure(case["stress_amplitude"], case["mean_stress"])
        sf = m.goodman_safety_factor(case["stress_amplitude"],
                                     case["mean_stress"])
        life = "infinite" if n == float("inf") else f"{n:9.3e}"
        print(f"  {case['label']:>8}: sigma_a={case['stress_amplitude']:5.0f} "
              f"sigma_m={case['mean_stress']:5.0f} n={case['cycles']:.1e} "
              f"-> N={life}, Goodman SF={sf:.2f}")

    damage = m.miner_damage()
    print(f"Miner damage per block:  D = {damage:.4f}")
    print(f"Blocks to failure:       {m.remaining_life():,.1f}")
    print()


def element_integration(m):
    """Use the Material inside any MechaElement (here: a Shaft)."""
    print("MechaElement Integration")
    print("=" * 40)

    shaft = Shaft(diameter=25, length=500, material=m, name="input shaft")
    props = shaft.material_properties  # SI (Pa) dict, as every element expects
    print(f"Shaft material: {shaft.material!r}")
    print(f"  yield_strength    = {props['yield_strength'] / 1e6:.0f} MPa")
    print(f"  ultimate_strength = {props['ultimate_strength'] / 1e6:.0f} MPa")
    print(f"  elastic_modulus   = {props['elastic_modulus'] / 1e9:.0f} GPa")

    # Inherited static yield check works unchanged (Pa in, ratio out)
    sf = shaft.safety_factor(125e6)  # 125 MPa applied
    print(f"Static SF at 125 MPa: {sf:.2f}")
    print()


def plots(m):
    """Save the Woehler and Goodman diagrams (requires matplotlib)."""
    print("Plots")
    print("=" * 40)
    try:
        fig_sn = m.plot_sn_curve(show=False)
        fig_goodman = m.plot_goodman(show=False)
    except ImportError as exc:
        print(f"skipped: {exc}")
        return
    fig_sn.savefig("material_sn_curve.png", dpi=150, bbox_inches="tight")
    fig_goodman.savefig("material_goodman.png", dpi=150, bbox_inches="tight")
    print("saved material_sn_curve.png and material_goodman.png")


if __name__ == "__main__":
    material = basics_and_marin()
    design_iteration(material)
    sn_curve_and_goodman(material)
    load_cases_and_miner(material)
    element_integration(material)
    plots(material)
