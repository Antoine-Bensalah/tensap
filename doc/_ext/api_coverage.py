"""Check that every public object of tensap appears in the API reference.

A public object is a class or a function of the ``tensap`` namespace whose
name does not start with an underscore and which is defined in tensap
itself (``ABC`` and ``abstractmethod``, imported from :mod:`abc`, are left
out). Each missing object raises a Sphinx warning, hence an error with
``-W``.
"""

import inspect
import re
from pathlib import Path

from sphinx.util import logging

logger = logging.getLogger(__name__)

_ENTRY = re.compile(r"^\s{3}([A-Za-z_][\w.]*)\s*$")


def _public_objects():
    import tensap

    names = set()
    for name in dir(tensap):
        if name.startswith("_"):
            continue
        obj = getattr(tensap, name)
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        if not getattr(obj, "__module__", "").startswith("tensap"):
            continue
        names.add(name)
    return names


def _documented_objects(api_dir):
    names = set()
    for path in Path(api_dir).glob("*.rst"):
        in_autosummary = False
        for line in path.read_text().splitlines():
            if line.startswith(".. autosummary::"):
                in_autosummary = True
                continue
            if in_autosummary and line and not line.startswith(" "):
                in_autosummary = False
            match = _ENTRY.match(line) if in_autosummary else None
            if match:
                names.add(match.group(1).split(".")[-1])
    return names


def check_api_coverage(app):
    api_dir = Path(app.srcdir) / "api"
    missing = sorted(_public_objects() - _documented_objects(api_dir))
    for name in missing:
        logger.warning(
            "tensap.%s is public but is not listed in doc/api/*.rst", name,
            type="api_coverage",
        )


def setup(app):
    app.connect("builder-inited", check_api_coverage)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
