"""Build a build123d model script, then export it or preview it.

    brepy [options] path/to/model.py

The script defines build(**parameters) returning {part name: shape}, where a
shape may instead be {shell name: shape}, the colour shells of one print. Each
part and shell gets a file, <name>_<part>[_<shell>], named after the script, and
the parts share one combined <name> file unless the script sets
SEPARATE_PRINTS = True. Output goes beside the script unless --out-dir says
otherwise; png writes preview sheets to its previews/ folder.

The script is imported by its path relative to --root, the current directory
by default, so it can import other modules under the root.

    brepy -f step -f png models/bracket/bracket.py
    brepy --watch models/bracket/bracket.py
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from brepy.build import load_model
from brepy.export import FORMATS, export


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brepy",
        description="Build a build123d model script, then export it or preview it.",
    )
    p.add_argument("model", type=Path, help="model script defining build()")
    p.add_argument(
        "-f",
        "--format",
        action="append",
        choices=FORMATS,
        dest="formats",
        help="export format, repeatable (default: step, unless previewing)",
    )
    p.add_argument("-o", "--out-dir", type=Path, help="output folder (default: the script's)")
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="folder the script is imported from (default: the current directory)",
    )
    p.add_argument("--part", help="export only this part")
    p.add_argument(
        "-p",
        "--param",
        action="append",
        default=[],
        dest="parameters",
        metavar="NAME=VALUE",
        help="argument for build(), repeatable",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="linear mesh tolerance in mm for stl and 3mf (default: 0.01)",
    )
    p.add_argument("--view", action="store_true", help="preview in a viewer window")
    p.add_argument(
        "--watch", action="store_true", help="preview, and rebuild whenever a source file changes"
    )
    p.add_argument("--port", type=int, default=3939, help="viewer port (default: 3939)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    path = args.model.resolve()
    root = args.root.resolve()
    previewing = args.view or args.watch
    formats = args.formats or ([] if previewing else ["step"])
    if formats:
        out_dir = (args.out_dir or args.model.parent).resolve()
        model = load_model(path, root)
        export(model, root, formats, out_dir, args.parameters, args.part, args.tolerance)
    if previewing:
        try:
            from brepy import viewer
        except ImportError as error:
            raise SystemExit(
                f"brepy: previewing needs the viewer extra, brepy[viewer] ({error})"
            ) from None
        session = viewer.Session(path, root, args.parameters, args.part, args.port)
        viewer.preview(session, args.watch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
