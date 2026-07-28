"""Generate and save visualizations of welded joint stress distributions."""

import matplotlib.pyplot as plt
from mecapy.welds import Weld, WeldedUnion, WeldLine, WeldCircle


def save_plots():
    """Generate and save stress distribution plots."""

    # Create figure with subplots for different examples
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Welded Joint Stress Distributions", fontsize=16, fontweight='bold')

    # ---- Example 1: T-Joint ----
    weld1 = Weld(
        path=WeldLine(start=(20, 0), end=(20, 80)),
        size=5,
        weld_type="fillet",
        electrode="E70xx"
    )
    weld2 = Weld(
        path=WeldLine(start=(80, 0), end=(80, 80)),
        size=5,
        weld_type="fillet",
        electrode="E70xx"
    )

    union1 = WeldedUnion(
        welds=[weld1, weld2],
        forces=(5000.0, 0.0, 8000.0),
        moments=(0.0, 0.0, 15000.0),
        name="T-Joint"
    )

    ax = axes[0, 0]
    union1.plot_distribution(n=24, show=False, ax=ax)
    ax.set_title("T-Joint: Shear + Axial + Torsion", fontweight='bold')

    # ---- Example 2: Box Joint ----
    box_welds = [
        Weld(WeldLine(start=(0, 0), end=(100, 0)), size=4, electrode="E70xx"),
        Weld(WeldLine(start=(100, 0), end=(100, 100)), size=4, electrode="E70xx"),
        Weld(WeldLine(start=(100, 100), end=(0, 100)), size=4, electrode="E70xx"),
        Weld(WeldLine(start=(0, 100), end=(0, 0)), size=4, electrode="E70xx"),
    ]

    union2 = WeldedUnion(
        welds=box_welds,
        forces=(2000.0, 3000.0, 5000.0),
        moments=(200000.0, 150000.0, 300000.0),
        name="Box Joint"
    )

    ax = axes[0, 1]
    union2.plot_distribution(n=24, show=False, ax=ax)
    ax.set_title("Box Joint: Complex 3D Loading", fontweight='bold')

    # ---- Example 3: Circular Weld ----
    circular_weld = Weld(
        path=WeldCircle(center=(50, 50), radius=40),
        size=6,
        weld_type="fillet",
        electrode="E70xx"
    )

    union3 = WeldedUnion(
        welds=[circular_weld],
        forces=(1000.0, 2000.0, 5000.0),
        moments=(100000.0, 50000.0, 200000.0),
    )

    ax = axes[1, 0]
    union3.plot_distribution(n=32, show=False, ax=ax)
    ax.set_title("Circular Weld: Torsion + Bending", fontweight='bold')

    # ---- Example 4: Cross Joint ----
    cross_welds = [
        Weld(WeldLine(start=(50, 0), end=(50, 100)), size=5, electrode="E70xx"),
        Weld(WeldLine(start=(0, 50), end=(100, 50)), size=5, electrode="E70xx"),
    ]

    union4 = WeldedUnion(
        welds=cross_welds,
        forces=(3000.0, 3000.0, 6000.0),
        moments=(150000.0, 150000.0, 250000.0),
    )

    ax = axes[1, 1]
    union4.plot_distribution(n=24, show=False, ax=ax)
    ax.set_title("Cross Joint: Symmetric Loading", fontweight='bold')

    #plt.tight_layout()
    #plt.show()
    #plt.savefig('welded_joint_stresses.png', dpi=150, bbox_inches='tight')
    #print("[OK] Saved: welded_joint_stresses.png")


if __name__ == "__main__":
    save_plots()
