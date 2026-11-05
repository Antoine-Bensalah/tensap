"""Numbered sections for the galleries and the API reference.

Theory & Methods is numbered by its own toctree option; the toctrees of the
Examples Gallery and of the API reference get the same, so that every
section of the site is numbered as in a book (3 levels: 2, 2.1, 2.1.3).
"""


def number_toctrees(app, docname, source):
    if docname in ("examples/index", "api/index"):
        source[0] = source[0].replace(
            ".. toctree::\n", ".. toctree::\n   :numbered: 3\n", 1
        )


def setup(app):
    app.connect("source-read", number_toctrees)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
