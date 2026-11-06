"""Drawn thumbnails for the cards of the Examples Gallery and the tutorials.

sphinx-gallery makes the thumbnail of an example from its last figure, or
uses a default image when there is none. Instead, each example of the
Examples Gallery has a simple drawing, in the colours of the site:
``doc/_thumbnails/<name>.png`` for ``examples/<section>/<name>.py``. The
drawings are made from the SVG files of the same folder; see its README.

After sphinx-gallery has generated the galleries, this extension copies
each drawing over the thumbnail of its example, and over the thumbnail of
the original tutorial the example comes from, and over the thumbnail of a
Use Case of the same name (``use_cases/<name>.py``) and of its original
tutorial. A tutorial can also have its own drawing, named after its script
(``tutorial_<...>.png``), which then takes precedence. The scripts are not
changed, so the gallery cache is not invalidated.
"""

import re
import shutil
from pathlib import Path

THUMB_DIR = "_thumbnails"
ORIGIN = re.compile(r"``test/non_regression_scripts/([^`]+)\.py``")


def _thumb(src, gallery, folder, stem):
    """Thumbnail written by sphinx-gallery for gallery/folder/stem.py."""
    images = src / gallery / folder / "images" / "thumb"
    return images / f"sphx_glr_{stem}_thumb.png"


def _own_drawing(src, script):
    """Whether the tutorial folder/stem has a drawing of its own."""
    stem = script.rpartition("/")[2]
    return (src / THUMB_DIR / f"{stem}.png").is_file()


def _targets(src):
    """Pairs (drawing, thumbnail to replace)."""
    repo = src.parent
    pairs = []
    for drawing in sorted((src / THUMB_DIR).glob("*.png")):
        name = drawing.stem
        for script in (repo / "examples").glob(f"*/{name}.py"):
            section = script.parent.name
            pairs.append((drawing, _thumb(src, "examples", section, name)))
            # The original tutorial named in the docstring of the example.
            match = ORIGIN.search(script.read_text(encoding="utf-8"))
            if match and not _own_drawing(src, match.group(1)):
                folder, _, stem = match.group(1).rpartition("/")
                thumb = _thumb(src, "tutorials", folder, stem)
                pairs.append((drawing, thumb))
        use_case = repo / "use_cases" / f"{name}.py"
        if use_case.is_file():
            images = src / "use_cases" / "images" / "thumb"
            pairs.append((drawing, images / f"sphx_glr_{name}_thumb.png"))
            # The original tutorial of the Use Case.
            match = ORIGIN.search(use_case.read_text(encoding="utf-8"))
            if match and not _own_drawing(src, match.group(1)):
                folder, _, stem = match.group(1).rpartition("/")
                pairs.append((drawing, _thumb(src, "tutorials", folder, stem)))
        # A tutorial with no example: the drawing has the name of the script.
        scripts = repo / "test" / "non_regression_scripts"
        for script in scripts.glob(f"*/{name}.py"):
            folder = script.parent.name
            pairs.append((drawing, _thumb(src, "tutorials", folder, name)))
    return pairs


def replace_thumbnails(app):
    for drawing, thumb in _targets(Path(app.srcdir)):
        if thumb.parent.is_dir():
            shutil.copyfile(drawing, thumb)


def setup(app):
    # After sphinx-gallery (priority 500), which writes its own thumbnails
    # when the galleries are generated.
    app.connect("builder-inited", replace_thumbnails, priority=900)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
