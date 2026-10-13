# brepy

Build, export and preview [build123d](https://build123d.readthedocs.io) model
scripts from the command line.

A model is a Python script with a `build()` function. `brepy` runs it, writes
the parts to STEP, STL, 3MF or BREP, draws dimensioned preview sheets without a
GPU or display, and can show the model in a live viewer that rebuilds on save.

## Install

```sh
uv tool install "brepy[viewer] @ git+https://github.com/brhelwig-dev/brepy"
```

The `viewer` extra adds `--view` and `--watch`. Leave it out on headless
machines. OCP needs the system library `libGL.so.1` (Debian: `libgl1`).

A container image with `brepy`, pytest, ruff and ty preinstalled is published
as `ghcr.io/brhelwig-dev/brepy:<version>` for each release, and as `:latest`,
rebuilt from `main` on every push and nightly:

```sh
docker run --rm -v "$PWD:/work" -w /work ghcr.io/brhelwig-dev/brepy:0.1.1 \
    brepy -f step models/bracket/bracket.py
```

## Usage

```sh
brepy models/bracket/bracket.py                     # STEP beside the script
brepy -f stl -f 3mf -o out/ models/bracket/bracket.py
brepy -f png models/bracket/bracket.py              # preview sheets
brepy --part lid models/box/box.py                  # one part
brepy -p width=30 models/bracket/bracket.py         # a build() argument
brepy --watch models/bracket/bracket.py             # live preview
```

The script is imported by its path relative to `--root`, the current directory
by default. `models/bracket/bracket.py` imports as `models.bracket.bracket`, so
scripts can share code by importing each other or helper modules under the
root.

## Model scripts

```python
from build123d import Box, Cylinder, Pos

SEPARATE_PRINTS = False


def build(width: float = 40.0) -> dict:
    base = Box(width, 20, 5)
    pin = Pos(0, 0, 5) * Cylinder(3, 10)
    return {"base": base, "pin": pin}


def preview_spec() -> dict:
    return {"parts": {"bracket_base": {"build_up": [0, 0, 1]}}}
```

`build()` returns parts by name. A part may itself be a mapping of shells,
`{"red": shape, "white": shape}`, for one print in several colours. Its
arguments can be set with `-p NAME=VALUE`, converted to the annotated type.

Output files are named after the script:

- one plain part: `<name>`
- several parts: `<name>_<part>` each, plus a combined `<name>` holding all of
  them, unless the script sets `SEPARATE_PRINTS = True`
- shells: `<name>_<part>_<shell>` each, plus `<name>_<part>` holding the shells

## Preview sheets

`-f png` writes one sheet per output file to `previews/`, with six orthographic
views (isometric, underside, front, right, top, bottom) at one shared scale.
`preview_spec()` adds to them:

```python
{
    "parts": {
        "<stem>": {
            "build_up": [0, 0, 1],
            "dimensions": [
                {"label": "pin dia", "view": "top", "a": [-3, 0, 15], "b": [3, 0, 15]},
                {"label": "slot", "view": "front", "a": [...], "b": [...], "detail": "slot"},
            ],
        }
    },
    "assemblies": {
        "<name>": {
            "seat": [
                {"stem": "<stem>", "at": [0, 0, 0], "build_up": [0, 0, 1]},
                {"stem": "../<other model>/<stem>"},
            ],
            "dimensions": [],
        }
    },
}
```

- `build_up` is the direction away from the build plate. With it, faces that
  would print without support are shaded orange.
- A dimension is drawn in the view it names, between points `a` and `b`. One
  with a `detail` name gets its own zoomed, sectioned tile.
- An assembly places output files, or files from a sibling script
  (`../<other model>/<other model>.py`), on one sheet.

When the spec is present, `-f step` also writes it to `preview.json`.

## Printability

`brepy.printability` measures a solid the same way the sheets shade it, for
use in a model's tests:

```python
from brepy import printability

assert printability.unsupported_area(part, build_up=(0, 0, 1)) < 0.01
assert printability.island_area(part, layer_height=0.2, build_up=(0, 0, 1)) < 0.01
```

A face is unsupported where it faces down more than 42 degrees from vertical
and does not rest on the plate. An island is an area printed in some layer with
nothing below it. `printability.section(part, z)` returns the plan section at
height `z`.

## Development

```sh
uv sync --extra viewer
uv run pytest
uv run ruff check && uv run ruff format --check
uv run ty check
```

## License

MIT
