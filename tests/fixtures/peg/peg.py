from build123d import Cylinder, Pos

RADIUS = 2.0


def build() -> dict:
    return {"peg": Pos(0, 0, 5) * Cylinder(RADIUS, 10)}
