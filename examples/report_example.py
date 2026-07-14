"""Example: collect elements and export a LaTeX report."""

from mecapy import Report
from mecapy.gears import Gear
from mecapy.shafts import Shaft
from mecapy.belts import Belt


def build_report(path="mecapy_report.tex"):
    """Build a small design report and save it as a .tex file."""
    pinion = Gear(teeth=17, module=2.5, face_width=38, name="pinion")
    shaft = Shaft(diameter=25.0, length=500.0, name="input shaft")
    belt = Belt(belt_type="v", friction=0.3, mass_per_length=0.4, name="drive belt")

    # A computed AGMA result to include alongside the components.
    bending = pinion.bending_stress(5000, 1200, geometry_factor=0.34)

    report = Report(title="Gearbox Design Report", author="P. Taboada")
    report.add_section("Transmission components")
    report.add(pinion, description="Driving pinion, 5 kW at 1200 rev/min.")
    report.add(shaft)
    report.add(belt)
    report.add_result(
        "AGMA gear check",
        [
            ("Bending stress", bending, "MPa"),
            ("Bending safety factor", pinion.bending_safety_factor(bending), ""),
        ],
    )

    saved = report.save(path)
    print(f"Report with {len(report)} components written to {saved}")
    return saved


if __name__ == "__main__":
    build_report()
