"""LaTeX report generation for MecaPy.

Collect the mechanical elements you define into a :class:`Report` and export
them as a compilable ``.tex`` document you can drop straight into a report.
"""

# LaTeX special-character escaping (order matters for the backslash).
_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

#: LaTeX end-of-row marker (a raw " \\").
ROW_END = r" \\"


def latex_escape(text):
    """Escape LaTeX special characters in free text/data.

    Labels and units defined inside the library are trusted and passed
    through verbatim; only user- or data-derived strings go through here.
    """
    return "".join(_ESCAPE.get(ch, ch) for ch in str(text))


def _format_float(value):
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e4 or magnitude < 1e-3:
        text = f"{value:.4g}"
        if "e" in text.lower():
            mantissa, exponent = text.lower().split("e")
            return f"{mantissa}\\times10^{{{int(exponent)}}}"
        return text
    return f"{value:g}"


def format_value(value):
    """Render a value as a LaTeX-ready cell (math-wrapped when numeric)."""
    if value is None:
        return "--"
    if isinstance(value, str):
        return latex_escape(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"${value}$"
    if isinstance(value, float):
        return "$" + _format_float(value) + "$"
    # sympy expressions, Fraction, etc.
    return latex_escape(str(value))


def render_table(rows):
    """Render ``(quantity, value, unit)`` rows as a booktabs tabular."""
    lines = [
        r"\begin{center}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Quantity & Value & Unit" + ROW_END,
        r"\midrule",
    ]
    for label, value, unit in rows:
        lines.append(f"{label} & {format_value(value)} & {unit}" + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def render_element(element, heading="subsection"):
    """Render a single element as a LaTeX heading plus a property table."""
    name = element.name if element.name else "(unnamed)"
    title = f"{element.element_type}: {latex_escape(name)}"
    return f"\\{heading}{{{title}}}\n" + render_table(element.latex_summary())


class Report:
    """
    A collector that turns mechanical elements into a LaTeX document.

    Add elements (and optional sections, notes and result tables) in order,
    then call :meth:`to_latex` for the document string or :meth:`save` to
    write a ``.tex`` file.

    Example:
        >>> from mecapy.beams import Beam
        >>> from mecapy.report import Report
        >>> report = Report(title="Gearbox design", author="Jane")
        >>> report.add(Beam(length=6.0, name="main shaft support"))
        <mecapy.report.Report object at ...>
        >>> path = report.save("report.tex")
    """

    def __init__(self, title="MecaPy Report", author=None, date=None):
        """
        Args:
            title (str): Document title.
            author (str): Optional author name.
            date (str): Optional date string. Defaults to LaTeX ``\\today``.
        """
        self.title = title
        self.author = author
        self.date = date
        self._items = []

    # -- building -------------------------------------------------------
    def add(self, element, description=None):
        """Add a mechanical element, with an optional description paragraph."""
        self._items.append(("element", element, description))
        return self

    def add_section(self, title):
        """Start a new top-level section."""
        self._items.append(("section", title, None))
        return self

    def add_note(self, text):
        """Add a free-text paragraph."""
        self._items.append(("note", text, None))
        return self

    def add_result(self, title, rows):
        """Add a standalone result table.

        Args:
            title (str): Subsection title.
            rows (list): List of ``(quantity, value, unit)`` tuples.
        """
        self._items.append(("result", title, list(rows)))
        return self

    @property
    def elements(self):
        """list: The elements added to the report, in order."""
        return [item[1] for item in self._items if item[0] == "element"]

    def __len__(self):
        return len(self.elements)

    # -- export ---------------------------------------------------------
    def to_latex(self):
        """Return the full LaTeX document as a string."""
        date = r"\today" if self.date is None else latex_escape(self.date)
        lines = [
            r"\documentclass[11pt]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{booktabs}",
            r"\usepackage{amsmath}",
            r"\usepackage[margin=2.5cm]{geometry}",
            r"\title{" + latex_escape(self.title) + "}",
            r"\author{" + latex_escape(self.author or "") + "}",
            r"\date{" + date + "}",
            r"\begin{document}",
            r"\maketitle",
        ]
        section_open = False
        for kind, payload, extra in self._items:
            if kind == "section":
                section_open = True
                lines.append(r"\section{" + latex_escape(payload) + "}")
            elif kind == "note":
                lines.append(latex_escape(payload))
            elif kind == "result":
                lines.append(r"\subsection{" + latex_escape(payload) + "}")
                lines.append(render_table(extra))
            elif kind == "element":
                if not section_open:
                    lines.append(r"\section{Components}")
                    section_open = True
                lines.append(render_element(payload))
                if extra:
                    lines.append(latex_escape(extra))
        lines.append(r"\end{document}")
        return "\n".join(lines) + "\n"

    def save(self, path):
        """Write the LaTeX document to ``path`` and return the path."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_latex())
        return path
