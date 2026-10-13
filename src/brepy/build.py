"""Load a model script and turn its parts into the files they are written to."""

import importlib
import inspect
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_type_hints

from build123d import Compound, Shape

Shapes = Mapping[str, Shape[Any]]
Parts = Mapping[str, Shape[Any] | Shapes]
Files = dict[str, Shapes]
Builder = Callable[..., Parts]
PreviewSpec = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class Model:
    name: str
    path: Path
    build: Builder
    preview_spec: PreviewSpec | None
    separate_prints: bool


def module_name(path: Path, root: Path) -> str:
    """The dotted name `path` imports as with `root` on sys.path."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise SystemExit(f"brepy: {path} is not under the root {root}") from None
    return ".".join(relative.with_suffix("").parts)


def model_from(path: Path, root: Path) -> Model:
    module = importlib.import_module(module_name(path, root))
    builder = getattr(module, "build", None)
    if not callable(builder):
        raise SystemExit(f"brepy: {path} defines no build()")
    preview_spec = getattr(module, "preview_spec", None)
    separate = bool(getattr(module, "SEPARATE_PRINTS", False))
    return Model(
        path.stem, path, builder, preview_spec if callable(preview_spec) else None, separate
    )


def project_file(file: str | None, root: Path) -> bool:
    """Whether `file` is the project's own source, not something installed under `root`."""
    if not file:
        return False
    path = Path(file).resolve()
    if not path.is_relative_to(root):
        return False
    parts = path.relative_to(root).parts
    return not any(p.startswith(".") or p == "site-packages" for p in parts)


def load_model(path: Path, root: Path) -> Model:
    """Import the script fresh, so a rebuild sees edits to it and to the modules it imports."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for name, module in list(sys.modules.items()):
        if name.partition(".")[0] != __package__ and project_file(
            getattr(module, "__file__", None), root
        ):
            del sys.modules[name]
    return model_from(path, root)


def parse_parameters(builder: Builder, assignments: Sequence[str]) -> dict[str, Any]:
    """NAME=VALUE strings converted to the types build() annotates."""
    hints = get_type_hints(builder)
    parameters: dict[str, Any] = {}
    for assignment in assignments:
        name, separator, text = assignment.partition("=")
        if not separator or name not in inspect.signature(builder).parameters:
            raise SystemExit(f"brepy: --param {assignment!r} is not NAME=VALUE for build()")
        kind = hints.get(name, str)
        try:
            parameters[name] = text.lower() in ("1", "true", "yes") if kind is bool else kind(text)
        except ValueError as error:
            raise SystemExit(f"brepy: --param {assignment!r}: {error}") from None
    return parameters


def output_files(parts: Parts, stem: str, separate_prints: bool) -> Files:
    """Each file to write, by stem: one per part and per colour shell, and one with
    every part unless they print separately. A lone plain part is just <stem>."""
    if len(parts) == 1:
        ((name, part),) = parts.items()
        if not isinstance(part, Mapping):
            part.label = name
            return {stem: {name: part}}
    files: Files = {}
    combined: dict[str, Shape[Any]] = {}
    for name, part in parts.items():
        if isinstance(part, Mapping):
            shells = {f"{name}_{shell}": shape for shell, shape in part.items()}
            files |= {f"{stem}_{label}": {label: shape} for label, shape in shells.items()}
        else:
            shells = {name: part}
        for label, shape in shells.items():
            shape.label = label
        files[f"{stem}_{name}"] = shells
        combined |= shells
    if not separate_prints:
        files[stem] = combined
    return files


def build_files(model: Model, parameters: Sequence[str], part: str | None) -> Files:
    """The files the model's parts go to, narrowed to `part`'s when it is given."""
    parts = model.build(**parse_parameters(model.build, parameters))
    files = output_files(parts, model.name, model.separate_prints)
    if part is None:
        return files
    prefix = f"{model.name}_{part}"
    chosen = {k: v for k, v in files.items() if k == prefix or k.startswith(f"{prefix}_")}
    if not chosen:
        raise SystemExit(f"brepy: no part {part!r}; parts are {', '.join(parts)}")
    return chosen


def combine(shapes: Shapes, label: str) -> Shape[Any]:
    """One shape holding every part as a distinct solid, each keeping its part name."""
    if len(shapes) == 1:
        return next(iter(shapes.values()))
    return Compound(children=list(shapes.values()), label=label)


def seat_built(model: Model, files: Files, root: Path) -> Callable[[str], Shape[Any]]:
    """An assembly's stem as a shape: one of these files, or ../<model>/<stem> from the
    sibling script ../<model>/<model>.py, built once."""
    siblings: dict[str, Files] = {}

    def seat(stem: str) -> Shape[Any]:
        if not stem.startswith("../"):
            return combine(files[stem], stem)
        name, _, sibling_stem = stem[3:].partition("/")
        if name not in siblings:
            path = (model.path.parent.parent / name / f"{name}.py").resolve()
            sibling = model_from(path, root)
            siblings[name] = output_files(sibling.build(), name, sibling.separate_prints)
        return combine(siblings[name][sibling_stem], sibling_stem)

    return seat
