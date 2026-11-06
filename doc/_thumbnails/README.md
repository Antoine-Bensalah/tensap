# Thumbnails of the Examples Gallery

One simple drawing per example of `examples/`, used on its card in the
Examples Gallery and on the card of the original tutorial it comes from, and
one per Use Case of `use_cases/`, which shows the test case and its equation
(see `doc/_ext/gallery_thumbnails.py`).

- `<name>.svg` is the drawing of `examples/<section>/<name>.py`: 400 x 280,
  light background `#f4f7fa`, strokes of 3 to 6 px, colours of the site
  (navy `#1f3a5f`, teal `#127a8a`, orange `#e8862a`, and their pale tints
  `#bfe0e5`, `#f6cfa8`, `#e9eff5`), no text in the examples; the Use
  Cases add the equation of the test case, in a serif italic font.
- `<name>.png` is the same drawing at twice the size (800 x 560), the file
  used by the build. Export it again after changing the SVG, for instance
  with `rsvg-convert -w 800 <name>.svg -o <name>.png`.

A new example without a drawing keeps the thumbnail made by sphinx-gallery
from its last figure.
