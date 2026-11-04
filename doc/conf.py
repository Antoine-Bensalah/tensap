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
    "sphinxcontrib.bibtex",
    "sphinx_design",  # cards and buttons of the home page
    "sphinx_gallery.gen_gallery",
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

# Figures, tables and code blocks are numbered, so that {numref} can cite them.
numfig = True

# -- Mathematics -------------------------------------------------------------

# The macros of math_notations.sty (used by the LaTeX manual and the PDF
# build), declared for MathJax so that the HTML pages use the same notations.
mathjax3_config = {
    "tex": {
        "macros": {
            **{c + "c": r"\mathcal{%s}" % c for c in "BHLMNPRTUYX"},
            **{c + "bb": r"\mathbb{%s}" % c for c in "RENP"},
            "argmax": r"\operatorname*{argmax}",
            "bigtimes": r"\mathop{\vcenter{\Large\times}}",
            "rank": r"\mathrm{rank}",
            "depth": r"\operatorname{depth}",
            "level": r"\operatorname{level}",
        }
    }
}

# -- Bibliography ------------------------------------------------------------

# One BibTeX file for the whole documentation, cited with {cite:p}`key` and
# listed on the bibliography page.
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

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
exclude_patterns = ["_build", "api_notes", "Thumbs.db", ".DS_Store", "**/*.ipynb"]

# -- Examples Gallery and Use cases (sphinx-gallery) -----------------------

# Examples that are not executed when the documentation is built: too long,
# or requiring tensorflow. Their code is shown without its output.
NOT_EXECUTED = [
    "tree_based_tensor_density_learning",
    "tensorized_function_learning",
    "tensor_completion",
    "digits_recognition",
]

sphinx_gallery_conf = {
    # Sources at the root of the repository, pages generated in doc/.
    "examples_dirs": ["../examples", "../use_cases"],
    "gallery_dirs": ["examples", "use_cases"],
    # Execute every example, except those of NOT_EXECUTED.
    "filename_pattern": r"/(?!(%s)\.py$)[^/]+\.py$" % "|".join(NOT_EXECUTED),
    "ignore_pattern": r"__init__\.py",
    # Sections and examples in the order of _ext/gallery_order.py.
    "subsection_order": [
        "../examples/tensors",
        "../examples/functions",
        "../examples/interpolation",
        "../examples/pca",
        "../examples/learning",
        "../examples/poincare",
    ],
    "within_subsection_order": "gallery_order.ExampleOrder",
    # Links from the code of the examples to the API reference, and lists
    # of examples at the bottom of each API page (see the class template).
    "doc_module": ("tensap",),
    "reference_url": {"tensap": None},
    "backreferences_dir": "api/backreferences",
    "remove_config_comments": True,
    "show_memory": False,
    "matplotlib_animations": False,
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"tensap {release}"
# Style of the site (colours of the mathematics and of the code).
html_static_path = ["_static"]
html_css_files = ["tensap.css"]

# Logo of tensap (an unbalanced tree tensor network), with a version for the
# dark theme, and the favicon (PNG for the browsers without SVG favicons).
html_favicon = "_static/tensap_favicon.png"

html_theme_options = {
    "logo": {
        "image_light": "_static/tensap_logo.svg",
        "image_dark": "_static/tensap_logo_dark.svg",
        "alt_text": "tensap",
    },
    "github_url": "https://github.com/anthony-nouy/tensap",
    "header_links_before_dropdown": 6,
    # Everything on one line in the header: the search is an icon (Ctrl+K
    # still opens it), next to the theme switcher and the GitHub link.
    "navbar_align": "left",
    "navbar_persistent": ["search-button"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_with_keys": False,
    "show_toc_level": 2,
    # Right column: the table of contents of the page only ("Show Source"
    # removed).
    "secondary_sidebar_items": ["page-toc"],
}
html_show_sourcelink = False
# Left column: the navigation of the section only (no "Collapse Sidebar"
# button).
html_sidebars = {"**": ["sidebar-nav-bs"]}
html_context = {"default_mode": "light"}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": r"\usepackage{math_notations}",
}
latex_additional_files = ["math_notations.sty"]
