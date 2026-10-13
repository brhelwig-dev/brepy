from build123d import Box, Cylinder, Pos

from peg import peg

HOLE = peg.RADIUS + 0.2


def build(width: float = 20.0, holed: bool = True) -> dict:
    base = Pos(0, 0, 2.5) * Box(width, 20, 5)
    if holed:
        base -= Pos(0, 0, 2.5) * Cylinder(HOLE, 5)
    tag = Pos(0, 0, 5.5) * Box(6, 3, 1)
    return {"base": base, "tag": {"red": tag, "white": Pos(0, 0, 6.5) * Box(4, 2, 1)}}


def preview_spec() -> dict:
    return {
        "parts": {
            "stand_base": {
                "build_up": [0, 0, 1],
                "dimensions": [
                    {"label": "hole dia", "view": "top", "a": [-HOLE, 0, 5], "b": [HOLE, 0, 5]}
                ],
            }
        },
        "assemblies": {
            "with_peg": {
                "seat": [
                    {"stem": "stand_base"},
                    {"stem": "../peg/peg", "at": [0, 0, 0], "build_up": [0, 0, 1]},
                ]
            }
        },
    }
