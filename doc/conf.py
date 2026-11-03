# Configuration file for the Sphinx documentation of tensap.
#
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html
# Editorial rules: see the Contributing section of the documentation.

import os
import sys

# Document the tensap of this repository, not an installed one.
sys.path.insert(0, os.path.abspath(".."))

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
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
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

autodoc_default_options = {"members": True, "inherited-members": True}
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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
