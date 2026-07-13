"""Sphinx configuration file for MecaPy documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "MecaPy"
copyright = "2025, Piter-T-UC"
author = "Piter-T-UC"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

language = "en"

exclude_patterns = ["_build"]

pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

htmlhelp_basename = "MecaPydoc"

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "12pt",
}

latex_documents = [
    (
        master_doc,
        "MecaPy.tex",
        "MecaPy Documentation",
        author,
        "manual",
    ),
]

man_pages = [
    (master_doc, "mecapy", "MecaPy Documentation", [author], 1),
]

texinfo_documents = [
    (
        master_doc,
        "MecaPy",
        "MecaPy Documentation",
        author,
        "MecaPy",
        "Python library for mechanical calculations",
        "Miscellaneous",
    ),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

todo_include_todos = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "show-inheritance": True,
}
