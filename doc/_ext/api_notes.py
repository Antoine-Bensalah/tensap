"""Insert mathematical notes, kept outside the code, to API docstrings.

For an object whose full name is ``tensap.FullTensor``, the file
``doc/api_notes/tensap.FullTensor.rst``, if it exists, is inserted in the
docstring under a "Mathematical description" rubric, after the extended
summary and before the first numpydoc section (Parameters, Attributes...). This keeps the
formulas out of the Python sources while showing them on the API page.

Only the page of the object itself gets the note, not the pages of its
methods or of its subclasses.
"""

from pathlib import Path

NOTES_DIR = "api_notes"
RUBRIC = "Mathematical description"


def _note_names(name, obj):
    # An object documented as tensap.FullTensor is defined in
    # tensap.tensor_algebra.tensors.full_tensor: accept both names.
    names = [name]
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module and qualname:
        names.append(f"{module}.{qualname}")
    return names


def _first_section(lines):
    # A numpydoc section is a title underlined with dashes.
    for i in range(len(lines) - 1):
        title, underline = lines[i].strip(), lines[i + 1].strip()
        if title and underline and set(underline) == {"-"} and len(underline) >= len(title):
            return i
    return len(lines)


def insert_note(app, what, name, obj, options, lines):
    if what not in ("class", "function"):
        return
    notes_dir = Path(app.srcdir) / NOTES_DIR
    for candidate in _note_names(name, obj):
        path = notes_dir / f"{candidate}.rst"
        if path.is_file():
            app.env.note_dependency(str(path))
            note = ["", f".. rubric:: {RUBRIC}", ""]
            note += path.read_text().splitlines() + [""]
            i = _first_section(lines)
            lines[i:i] = note
            return


def setup(app):
    # Priority 400: before numpydoc (500), which then reads the note as part
    # of the extended summary.
    app.connect("autodoc-process-docstring", insert_note, priority=400)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
