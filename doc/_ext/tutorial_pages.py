"""The "Tensap tutorials", a subsection of the Use cases.

The original tutorials of tensap are kept, unchanged, as non-regression
scripts in ``test/non_regression_scripts/``. When the documentation is
built, this extension copies each script into ``doc/_tutorials_src/`` (not
versioned), as a sphinx-gallery example:

- a docstring with a readable title (TITLES below) replaces the original
  one, whose first sentence is kept as the summary;
- each cell ``# %% Title`` of the script becomes a section ``Title``, so
  that the page follows the cells of the script;
- the code itself is unchanged.

sphinx-gallery then executes the copies (except those of NOT_EXECUTED in
conf.py), shows their outputs and figures, caches them, and offers the
script for download: it runs cell by cell as the original does.

The Use cases list the tutorials last, after the use cases. The sections
of the Examples Gallery, of the Use cases, of the tutorials and of the API
reference are numbered, as in Theory & Methods.
"""

import ast
import re
from pathlib import Path

SRC_DIR = "_tutorials_src"
GALLERY_DIR = "tutorials"
FOLDERS = {
    "tensor_algebra": "Tensor algebra",
    "functions": "Functions",
    "approximation": "Approximation",
    "poincare_learning": "Poincaré learning",
    "tools": "Tools",
}
TITLES = {
    "tutorial_FullTensor": "Full tensors",
    "tutorial_DimensionTree": "Dimension trees",
    "tutorial_TreeBasedTensor": "Tree-based tensors",
    "tutorial_tensor_truncation": "Truncation of tensors",
    "tutorial_measures": "Measures and random variables",
    "tutorial_orthonormal_polynomials": "Orthonormal polynomials",
    "tutorial_functions_bases_grids": "Functions, bases and grids",
    "tutorial_conditional_expectation": "Conditional expectations",
    "tutorial_TensorizedFunction": "Tensorized functions",
    "tutorial_BSplines": "B-splines",
    "tutorial_piecewise_polynomials": "Piecewise polynomials",
    "tutorial_integration_rules": "Integration rules",
    "tutorial_FunctionalBasis_interpolation_projection_approximation":
        "Interpolation, projection and least squares on a functional basis",
    "tutorial_magic_points": "Magic points and interpolation",
    "tutorial_interpolation_Tensor": "Interpolation in tensor format",
    "tutorial_leastsquares_Tensor": "Least squares in tensor format",
    "tutorial_PCA_TensorPrincipalComponentAnalysis":
        "Principal component analysis of tensors",
    "tutorial_PCA_FunctionalTensorPrincipalComponentAnalysis":
        "Functional principal component analysis",
    "tutorial_TensorizedFunction_Learning_PCA":
        "Tensorized functions and functional PCA",
    "tutorial_tensor_learning_CanonicalTensorLearning":
        "Learning in canonical tensor format",
    "tutorial_tensor_learning_TreeBasedTensorLearning":
        "Learning in tree-based tensor format",
    "tutorial_tensor_learning_TreeBasedTensorDensityLearning":
        "Density estimation in tree-based tensor format",
    "tutorial_tensor_learning_tensorized_function_learning":
        "Learning a tensorized function",
    "tutorial_tensor_learning_tensor_completion": "Tensor completion",
    "tutorial_tensor_learning_digits_recognition": "Digits recognition",
    "tutorial_poincare_learning": "Poincaré learning of nonlinear features",
    "tutorial_poincare_learning_surrogate":
        "Poincaré learning with a convex surrogate",
    "tutorial_poincare_learning_surrogate_greedy":
        "Greedy learning of several features",
    "tutorial_poincare_learning_surrogate_collective":
        "Collective Poincaré learning",
    "tutorial_MultiIndices": "Multi-indices",
}
ORIGIN = re.compile(r"``test/non_regression_scripts/([^`]+\.py)``")
CELL = re.compile(r"^# %%\s*(.*?)\s*$")

INDEX = """\
.. _tensap-tutorials:

Tensap tutorials
================

The original tutorials of tensap, as they are in the repository
(``test/non_regression_scripts/``, formerly ``tutorials/``), executed and
shown cell by cell. Each one can be downloaded as a script and run again,
cell by cell. Most of them are also in the :doc:`/examples/index`, rewritten
as commented steps.
"""


def _gallery_links(repo):
    """Map each script to the label of its rewritten page in the gallery."""
    links = {}
    for path in sorted((repo / "examples").glob("*/*.py")):
        match = ORIGIN.search(path.read_text(encoding="utf-8"))
        if match:
            section, name = path.parent.name, path.stem
            links[match.group(1)] = f"sphx_glr_examples_{section}_{name}.py"
    return links


def _split_docstring(text):
    """Header comments, first sentence of the docstring, code after it."""
    tree = ast.parse(text)
    lines = text.splitlines()
    first = tree.body[0] if tree.body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        doc = ast.get_docstring(tree)
        summary = " ".join(doc.split("\n\n")[0].split())
        header = lines[: first.lineno - 1]
        rest = lines[first.end_lineno:]
    else:
        # No docstring: the license header is the leading block of comments.
        n = 0
        while n < len(lines) and (lines[n].startswith("#") or not lines[n].strip()):
            if CELL.match(lines[n]):
                break
            n += 1
        summary, header, rest = None, lines[:n], lines[n:]
    return header, summary, rest


def _convert(script, rel, link):
    header, summary, rest = _split_docstring(script.read_text(encoding="utf-8"))
    title = TITLES.get(script.stem, script.stem)
    doc = [title, "=" * len(title), ""]
    if summary:
        doc += [summary, ""]
    doc.append(f"Original script: ``test/non_regression_scripts/{rel}``.")
    if link:
        doc[-1] += f" Rewritten as commented steps in the gallery: :ref:`{link}`."
    out = ['"""', *doc, '"""', ""]
    # The license header, as comments of the first block of code, shortened
    # to its SPDX identifier as in the examples of the gallery.
    header = [line for line in header if line.strip()]
    if any("GNU Lesser General Public License" in line for line in header):
        header = header[:1] + ["# SPDX-License-Identifier: LGPL-3.0-or-later"]
    out += header + ([""] if header else [])
    for line in rest:
        if re.fullmatch(r"#{20,}\s*", line):
            # A line of hashes is also a separator of blocks for
            # sphinx-gallery: the cells are enough.
            continue
        match = CELL.match(line)
        if match and match.group(1):
            name = match.group(1)
            out += ["# %%", f"# {name}", f"# {'-' * len(name)}", "#"]
        else:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


def _write(path, text):
    # Do not touch an unchanged file, so that sphinx-gallery does not run
    # the example again.
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def generate(srcdir):
    src = Path(srcdir)
    repo = src.parent
    scripts_dir = repo / "test" / "non_regression_scripts"
    out = src / SRC_DIR
    links = _gallery_links(repo)
    _write(out / "README.txt", INDEX)
    for folder, title in FOLDERS.items():
        scripts = sorted((scripts_dir / folder).glob("*.py"))
        if not scripts:
            continue
        _write(out / folder / "README.txt", f"{title}\n{'-' * len(title)}\n")
        for script in scripts:
            rel = f"{folder}/{script.name}"
            _write(out / folder / script.name, _convert(script, rel, links.get(rel)))


def number_toctrees(app, docname, source):
    """Number the sections of the gallery, of the use cases and of the API."""
    text = source[0]
    if docname in ("examples/index", "api/index"):
        text = text.replace(".. toctree::\n", ".. toctree::\n   :numbered: 3\n", 1)
    elif docname == "use_cases/index":
        # The tutorials come last, in the same table of contents, numbered
        # one level deeper (4 Tensap tutorials, 4.1 Tensor algebra, 4.1.2 a
        # tutorial, 4.1.2.1 its cells).
        text = text.replace(".. toctree::\n", ".. toctree::\n   :numbered: 4\n", 1)
        text = re.sub(
            r"(\.\. toctree::\n(?:   :.*\n)*\n(?:   \S.*\n)+)",
            lambda m: m.group(1) + "   /tutorials/index\n",
            text,
            count=1,
        )
    elif docname == "tutorials/index":
        # sphinx-gallery puts the table of contents at the end of the page,
        # after the sections of its folders: its entries would be numbered
        # under the last section. Before the first section, they are
        # numbered as the sections themselves.
        toctree = re.search(r"\n\.\. toctree::\n(?:   .*\n|\n)+", text)
        if toctree:
            block = toctree.group(0)
            text = text.replace(block, "\n", 1)
            text = text.replace("\n.. raw:: html", block + "\n.. raw:: html", 1)
        # The headings of the folders on this page would be numbered after
        # the entries of the table of contents (4.6 for 4.1): they are shown
        # as rubrics, styled as headings, without number.
        for title in FOLDERS.values():
            text = text.replace(
                f"\n{title}\n{'-' * len(title)}\n",
                f"\n.. rubric:: {title}\n   :class: tutorial-folder\n",
            )
    source[0] = text


def setup(app):
    # The sources must exist before sphinx-gallery looks for them.
    generate(app.srcdir)
    app.connect("source-read", number_toctrees)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
