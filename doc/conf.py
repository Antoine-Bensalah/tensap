# Configuration file for the Sphinx documentation of tensap.
#
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html
# Editorial rules: see the Contributing section of the documentation.

import os
import sys

# Document the tensap of this repository, not an installed one.
sys.path.insert(0, os.path.abspath(".."))
# Local Sphinx extensions of this documentation.
sys.path.insert(0, os.path.abspath("_ext"))

import tensap  # noqa: E402

# -- Project information -----------------------------------------------------

project = "tensap"
author = "A. Nouy, E. Grelier"
copyright = "2020-2026, A. Nouy, E. Grelier and contributors"
# Read from the package, so that it cannot drift from pyproject.toml.
release = tensap.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
    "api_coverage",  # _ext/: every public object is in doc/api/
    "api_notes",  # _ext/: formulas of doc/api_notes/ appended to docstrings
]

# Narrative pages are written in MyST Markdown (.md); the API pages and the
# docstrings stay in reStructuredText (.rst), as numpydoc expects.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# MyST: write mathematics as in LaTeX ($...$, $$...$$, amsmath environments).
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# -- API reference -------------------------------------------------------------

# One page per public object, generated in doc/api/generated/ (not versioned)
# from the lists of doc/api/*.rst and the templates of _templates/autosummary.
autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True}
autodoc_typehints = "none"
numpydoc_show_class_members = True
# Minimum required of every public object (see doc_technical_choices.md):
# GL08, a docstring. Methods are not checked yet (pattern below); the other
# checks (summary, Parameters, Returns...) will be added sub-package by
# sub-package. A failed check is a warning, hence an error with -W.
numpydoc_validation_checks = {"GL08"}
numpydoc_validation_exclude = {r"^tensap\.\w+\.\w+$"}
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
# api_notes/ holds fragments included by _ext/api_notes.py, not pages.
exclude_patterns = ["_build", "api_notes", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"tensap {release}"
html_theme_options = {
    "github_url": "https://github.com/anthony-nouy/tensap",
    "header_links_before_dropdown": 6,
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_context = {"default_mode": "light"}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": r"\usepackage{math_notations}",
}
latex_additional_files = ["math_notations.sty"]
