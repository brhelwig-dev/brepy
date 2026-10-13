"""Write a model's files in the requested formats, and its preview sheets."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from build123d import Mesher, export_brep, export_step, export_stl

from brepy import sheets
from brepy.build import Model, Shapes, build_files, combine, seat_built

FORMATS = ("step", "stl", "3mf", "brep", "png")


def write(shapes: Shapes, label: str, path: Path, tolerance: float) -> None:
    match path.suffix:
        case ".step":
            export_step(combine(shapes, label), str(path))
        case ".stl":
            export_stl(combine(shapes, label), str(path), tolerance=tolerance)
        case ".brep":
            export_brep(combine(shapes, label), str(path))
        case ".3mf":
            mesher = Mesher()
            for shape in shapes.values():
                mesher.add_shape(shape, linear_deflection=tolerance)
            mesher.write(str(path))
        case _:
            raise SystemExit(f"brepy: unknown output format {path.suffix}")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def export(
    model: Model,
    root: Path,
    formats: Sequence[str],
    out_dir: Path,
    parameters: Sequence[str] = (),
    part: str | None = None,
    tolerance: float = 0.01,
) -> None:
    files = build_files(model, parameters, part)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt != "png":
            for stem, shapes in files.items():
                write(shapes, stem, out_dir / f"{stem}.{fmt}", tolerance)
    spec: dict[str, Any] = {}
    if model.preview_spec is not None and {"png", "step"} & set(formats):
        spec = model.preview_spec()
    if "png" in formats:
        shapes = {stem: combine(held, stem) for stem, held in files.items()}
        sheets.draw_sheets(shapes, spec, seat_built(model, files, root), out_dir / "previews")
    if spec and "step" in formats:
        path = out_dir / "preview.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"wrote {path}")
