"""Printability measurements on a built solid, for a given build direction.

`build_up` points away from the plate. A face is unsupported where it points
down past 42 degrees from vertical and is not the layer on the plate, the rule
the preview sheets shade orange by.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from build123d import Compound, Face, Pos, Sketch, Solid, Vector
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_State

from brepy import sheets

Body = Solid | Compound


def unsupported_triangles(
    shape: Body, build_up: Sequence[float]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Area and centroid of each unsupported triangle. The mesh finds candidates as the
    sheets do; each is then judged by the surface's own normal at its centroid, so a
    sliver facet along a boolean seam cannot read as an overhang."""
    up = sheets.unit(build_up)
    areas, centroids = [], []
    for solid in shape.solids():
        classifier = BRepClass3d_SolidClassifier(solid.wrapped)
        meshes = []
        for face in solid.faces():
            if face.area > sheets.SLIVER_AREA:
                meshes.append((face, *sheets.face_mesh(face)))
        floor = min((v @ up).min() for _, v, t in meshes if len(t))
        for face, vertices, triangles in meshes:
            if not len(triangles):
                continue
            down = sheets.plate_ward(vertices, triangles, classifier, up)
            down &= ~sheets.resting_on(vertices, triangles, up, floor)
            for a, b, c in vertices[triangles[down]]:
                centre = (a + b + c) / 3
                if surface_points_down(face, centre, classifier, up):
                    areas.append(float(np.linalg.norm(np.cross(b - a, c - a))) / 2)
                    centroids.append(centre)
    return np.array(areas), np.array(centroids).reshape(-1, 3)


def surface_points_down(
    face: Face, point: npt.NDArray[np.float64], classifier: Any, up: npt.NDArray[np.float64]
) -> bool:
    """Whether the face's outward normal at `point` leans down past the overhang limit."""
    normal = np.array(tuple(face.normal_at(Vector(*point))))
    classifier.Perform(gp_Pnt(*(point + normal * sheets.PROBE_STEP)), 1e-6)
    if classifier.State() == TopAbs_State.TopAbs_IN or classifier.IsOnAFace():
        normal = -normal
    return bool(normal @ up < -sheets.OVERHANG_COS)


def unsupported_area(shape: Body, build_up: Sequence[float]) -> float:
    return float(unsupported_triangles(shape, build_up)[0].sum())


def section(shape: Body, z: float) -> Sketch:
    """The plan section of `shape` at height z, dropped onto z=0."""
    cut = shape & (Pos(0, 0, z) * Face.make_rect(1000, 1000))
    return Pos(0, 0, -z) * Sketch(cut.faces()) if cut is not None else Sketch()


def island_area(shape: Body, layer_height: float, build_up: Sequence[float]) -> float:
    """The largest area printed in some layer with nothing at all under it, walking the
    layers in print order along Z."""
    bounds = shape.bounding_box()
    step = layer_height if build_up[2] > 0 else -layer_height
    z = bounds.min.Z + layer_height / 2 if step > 0 else bounds.max.Z - layer_height / 2
    worst, below = 0.0, None
    while bounds.min.Z < z < bounds.max.Z:
        layer = section(shape, z)
        if below is not None and layer.faces():
            for face in (layer - below).faces():
                if face.distance_to(below) > 1e-6:
                    worst = max(worst, face.area)
        if layer.faces():
            below = layer
        z += step
    return worst
