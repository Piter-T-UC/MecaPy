"""Tests for LaTeX report generation."""

import pytest

from mecapy import Report, Material
from mecapy.beams import Beam
from mecapy.gears import Gear
from mecapy.shafts import Shaft
from mecapy import report as report_mod


class TestLatexHelpers:
    def test_escape_special_characters(self):
        assert report_mod.latex_escape("a_b") == "a\\_b"
        assert report_mod.latex_escape("50%") == "50\\%"
        assert report_mod.latex_escape("x&y#z") == "x\\&y\\#z"

    def test_format_int(self):
        assert report_mod.format_value(17) == "$17$"

    def test_format_float_scientific(self):
        assert report_mod.format_value(210e9) == "$2.1\\times10^{11}$"

    def test_format_float_plain(self):
        assert report_mod.format_value(2.5) == "$2.5$"

    def test_format_string_escaped(self):
        assert report_mod.format_value("cast_iron") == "cast\\_iron"

    def test_format_none(self):
        assert report_mod.format_value(None) == "--"


class TestElementLatex:
    def test_element_to_latex_structure(self):
        beam = Beam(length=6.0, second_moment=8e-6, name="beam1")
        tex = beam.to_latex()
        assert r"\subsection{Beam: beam1}" in tex
        assert r"\begin{tabular}{lll}" in tex
        assert "Length" in tex and "Material" in tex

    def test_element_type(self):
        assert Gear(teeth=20, module=2.5).element_type == "Gear"

    def test_name_is_escaped(self):
        beam = Beam(length=1.0, name="main_beam")
        assert "main\\_beam" in beam.to_latex()

    def test_material_instance_name(self):
        mat = Material("titanium", elastic_modulus=114e9, poisson_ratio=0.34)
        gear = Gear(teeth=18, module=2.0, material=mat)
        rows = gear.latex_summary()
        assert ("Material", "titanium", "") in rows

    def test_callable_field_is_evaluated(self):
        # Shaft.polar_moment is a property -> value appears, not a method repr.
        shaft = Shaft(diameter=20.0, length=500.0)
        rows = dict((label, value) for label, value, _ in shaft.latex_summary())
        assert isinstance(rows["Polar moment $J$"], float)


class TestReport:
    def test_add_returns_self_and_len(self):
        report = Report()
        assert report.add(Beam(length=1.0)) is report
        assert len(report) == 1

    def test_to_latex_is_complete_document(self):
        report = Report(title="My Report", author="Tester")
        report.add(Gear(teeth=20, module=2.5, name="g1"))
        tex = report.to_latex()
        assert tex.startswith(r"\documentclass")
        assert r"\begin{document}" in tex
        assert r"\end{document}" in tex
        assert r"\title{My Report}" in tex
        assert "Gear: g1" in tex

    def test_auto_components_section(self):
        report = Report()
        report.add(Beam(length=1.0, name="b"))
        assert r"\section{Components}" in report.to_latex()

    def test_add_result_table(self):
        report = Report()
        report.add_result("Results", [("Stress", 75.9, "MPa")])
        tex = report.to_latex()
        assert r"\subsection{Results}" in tex
        assert "Stress" in tex and "MPa" in tex

    def test_save_writes_file(self, tmp_path):
        report = Report(title="Saved")
        report.add(Beam(length=2.0, name="beam"))
        path = report.save(str(tmp_path / "out.tex"))
        content = open(path, encoding="utf-8").read()
        assert r"\end{document}" in content

    def test_title_with_special_chars_escaped(self):
        report = Report(title="Load & Stress 100%")
        assert r"Load \& Stress 100\%" in report.to_latex()
