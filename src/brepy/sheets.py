"""Preview sheets: six dimensioned views of a part, drawn without a GPU or display.

Each solid is tessellated and projected orthographically, filled through a
z-buffer with flat shading, and overdrawn with its B-rep edges. brepy calls
draw_sheets() with the shapes it exports and the model's preview_spec():

    {"parts": {"<stem>": {"build_up": [0, 0, 1],
                          "dimensions": [{"label": "pin hole dia", "view": "right",
                                          "a": [x, y, z], "b": [x, y, z],
                                          "detail": "<optional zoomed tile>"}]}},
     "assemblies": {"<name>": {"seat": [{"stem": "<stem or ../<model>/<stem>>",
                                        "at": [0, 0, 0], "build_up": [0, 0, 1]}],
                               "dimensions": [...]}}}

`build_up` points away from the build plate; faces that print unsupported are
drawn orange. A dimension draws the distance between its two points, and is
skipped in a view it is not parallel to.
"""

import itertools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from build123d import Compound, Face, Pos, Shape, Solid
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_Orientation, TopAbs_State
from OCP.TopLoc import TopLoc_Location
from PIL import Image, ImageDraw, ImageFont

Array = npt.NDArray[np.float64]
Box2 = tuple[float, float, float, float]
Dimension = Mapping[str, Any]
Ints = npt.NDArray[np.int64]
Group = tuple[Array, Ints, npt.NDArray[np.bool_]]
Projector = Callable[[Any], tuple[Array, Array, Array]]
Colour = tuple[int, int, int]
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

SS: Final = 2
TILE: Final = (960, 720)
COLS: Final = 2
MARGIN: Final = 0.06
TESS_TOL: Final = 0.05
SLIVER_AREA: Final = 1e-6

BG: Final[Colour] = (255, 255, 255)
INK: Final[Colour] = (40, 44, 48)
DIM: Final[Colour] = (46, 96, 158)
BODY: Final[Colour] = (176, 182, 189)
UNSUPPORTED: Final[Colour] = (214, 122, 92)

OVERHANG_COS: Final = float(np.cos(np.radians(42.0)))
PROBE_STEP: Final = 0.2

U: Final = SS * TILE[1] / 360.0
FONT_PX: Final = int(12 * U)
LW: Final = max(1, int(U))
GAP: Final = int(4 * U)
OVER: Final = int(5 * U)
OFF: Final = int(18 * U)
FEAT_OFF: Final = int(14 * U)
BAND: Final = int(68 * U)
ARROW_L: Final = int(8 * U)
ARROW_W: Final = int(3 * U)
DETAIL_PAD: Final = 3.0

VIEWS: Final = (
    ("isometric", (-1.0, 1.0, -1.0), (0.0, 0.0, 1.0)),
    ("underside", (-1.0, 1.0, 1.0), (0.0, 0.0, -1.0)),
    ("front", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ("right", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("top", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ("bottom", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
)


def unit(v: Any) -> Array:
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


def basis(direction: Sequence[float], up_hint: Sequence[float]) -> tuple[Array, Array, Array]:
    """Orthonormal camera axes (right, up, forward)."""
    fwd = unit(direction)
    right = np.cross(fwd, unit(up_hint))
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(fwd, (1.0, 0.0, 0.0))
    right = unit(right)
    return right, unit(np.cross(right, fwd)), fwd


def axis_of(vec: Array) -> int | None:
    """The world axis a camera vector lies along, or None."""
    a = np.abs(vec)
    i = int(np.argmax(a))
    return i if a[i] > 0.999 else None


def tessellate(shape: Shape[Any], build_up: Sequence[float] | None) -> tuple[list[Group], set[int]]:
    """Per solid: vertices, triangles and which triangles print unsupported;
    and the hashes of the edges bounding a face that carries any of those."""
    up = unit(build_up) if build_up is not None else None
    groups: list[Group] = []
    flagged: set[int] = set()
    for solid in shape.solids():
        meshes = []
        for face in solid.faces():
            if face.area < SLIVER_AREA:
                continue
            vs, ts = face_mesh(face)
            if len(ts):
                meshes.append((face, vs, ts))
        if not meshes:
            continue
        floor = min((v @ up).min() for _, v, _ in meshes) if up is not None else 0.0
        classifier = BRepClass3d_SolidClassifier(solid.wrapped)
        verts, tris, down, base = [], [], [], 0
        for face, v, t in meshes:
            d = plate_ward(v, t, classifier, up) & ~resting_on(v, t, up, floor)
            if d.any():
                flagged.update(hash(e) for e in face.edges())
            verts.append(v)
            tris.append(t + base)
            down.append(d)
            base += len(v)
        groups.append((np.vstack(verts), np.vstack(tris), np.concatenate(down)))
    return groups, flagged


def face_mesh(face: Face) -> tuple[Array, Ints]:
    """The face's vertices and triangles, as face.tessellate(TESS_TOL) gives them."""
    face.mesh(TESS_TOL)
    loc = TopLoc_Location()
    poly = BRep_Tool.Triangulation_s(face.wrapped, loc)
    if poly is None:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    nodes = (poly.Node(i) for i in range(1, poly.NbNodes() + 1))
    if not loc.IsIdentity():
        trsf = loc.Transformation()
        nodes = (p.Transformed(trsf) for p in nodes)
    verts = np.array([(p.X(), p.Y(), p.Z()) for p in nodes]).reshape(-1, 3)
    order = (
        (1, 3, 2) if face.wrapped.Orientation() == TopAbs_Orientation.TopAbs_REVERSED else (1, 2, 3)
    )
    tris = np.array(
        [tuple(t.Value(k) - 1 for k in order) for t in poly.Triangles()], dtype=np.int64
    ).reshape(-1, 3)
    return verts, tris


def plate_ward(verts: Array, tris: Any, classifier: Any, up: Array | None) -> npt.NDArray[np.bool_]:
    """The triangles of one face that lean toward the plate past the overhang limit.
    Winding does not say which side is material, so the solid is probed once per face."""
    none = np.zeros(len(tris), dtype=bool)
    n = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    lengths = np.linalg.norm(n, axis=1)
    ok = lengths > 1e-12
    if up is None or not ok.any():
        return none
    n[ok] /= lengths[ok][:, None]
    if (np.abs(n[ok] @ up) < OVERHANG_COS).all():
        return none
    k = int(np.argmax(lengths))
    probe = verts[tris[k]].mean(axis=0) + n[k] * PROBE_STEP
    classifier.Perform(gp_Pnt(*probe), 1e-6)
    if classifier.State() == TopAbs_State.TopAbs_IN or classifier.IsOnAFace():
        n = -n
    return ok & ((n @ up) < -OVERHANG_COS)


def resting_on(verts: Array, tris: Any, up: Array | None, floor: float) -> npt.NDArray[np.bool_]:
    """Triangles lying on the plate, the one downward surface that is supported."""
    if up is None:
        return np.zeros(len(tris), dtype=bool)
    return ((verts @ up)[tris] - floor < TESS_TOL).all(axis=1)


def drawing_centre(size: tuple[int, int]) -> tuple[float, float]:
    """Centre of the tile less the dimension band."""
    return (size[0] - BAND) / 2.0, (size[1] - BAND) / 2.0


def projector(
    right: Array, up: Array, fwd: Array, size: tuple[int, int], scale: float, centre: Array
) -> Projector:
    """World points (N, 3) -> pixel x, pixel y, depth."""
    ox, oy = drawing_centre(size)

    def to_px(points: Any) -> tuple[Array, Array, Array]:
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        return (p @ right - centre[0]) * scale + ox, oy - (p @ up - centre[1]) * scale, p @ fwd

    return to_px


def fit(lo: Array, hi: Array, size: tuple[int, int]) -> tuple[Array, float]:
    """Centre and scale that fit the projected span lo..hi in the drawing area."""
    extent = np.maximum(hi - lo, 1e-6)
    ox, oy = drawing_centre(size)
    scale = min(2 * ox * (1 - 2 * MARGIN) / extent[0], 2 * oy * (1 - 2 * MARGIN) / extent[1])
    return (lo + hi) / 2.0, scale


def view_frame(
    groups: list[Group], direction: Sequence[float], up_hint: Sequence[float], size: tuple[int, int]
) -> tuple[Array, float]:
    right, up, _ = basis(direction, up_hint)
    xy = np.vstack([np.column_stack([v @ right, v @ up]) for v, _, _ in groups])
    return fit(xy.min(axis=0), xy.max(axis=0), size)


def section(shape: Shape[Any], fwd: Array, depth: float) -> Shape[Any]:
    """`shape` with everything nearer the camera than `depth` cut away."""
    axis = axis_of(fwd)
    if axis is None:
        raise ValueError("a detail must look along an axis to section cleanly")
    bb = shape.bounding_box()
    lo = np.array(tuple(bb.min)) - 10.0
    hi = np.array(tuple(bb.max)) + 10.0
    if fwd[axis] > 0:
        hi[axis] = depth * fwd[axis]
    else:
        lo[axis] = depth * fwd[axis]
    return shape - Pos(*lo) * Solid.make_box(*(hi - lo))


def rasterise(img: Array, zbuf: Array, groups: list[Group], to_px: Projector, light: Array) -> None:
    """Fill every triangle through the z-buffer, shaded on |n.L| so winding does not matter.
    Where triangles tie on depth the first one drawn keeps the pixel."""
    for verts, tris, unsupported in groups:
        px, py, depth = to_px(verts)
        normals = np.cross(
            verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]
        )
        lengths = np.linalg.norm(normals, axis=1)
        ok = lengths > 1e-12
        normals[ok] /= lengths[ok][:, None]
        shade = 0.35 + 0.65 * np.abs(normals @ light)
        colour = np.where(unsupported[:, None], UNSUPPORTED, BODY) * shade[:, None]
        x, y, z = px[tris[ok]], py[tris[ok]], depth[tris[ok]]
        fill(img, zbuf, x, y, z, colour[ok])


FRAGMENTS: Final = 1 << 20  # pixels tested per batch of triangles


def fill(img: Array, zbuf: Array, x: Array, y: Array, z: Array, colour: Array) -> None:
    """Fill triangles (N, 3 corners) in order, each pixel kept by the nearest fragment."""
    h, w = zbuf.shape
    x0 = np.maximum(np.floor(x.min(axis=1)).astype(np.int64), 0)
    x1 = np.minimum(np.ceil(x.max(axis=1)).astype(np.int64) + 1, w)
    y0 = np.maximum(np.floor(y.min(axis=1)).astype(np.int64), 0)
    y1 = np.minimum(np.ceil(y.max(axis=1)).astype(np.int64) + 1, h)
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    drawn = np.flatnonzero((x1 > x0) & (y1 > y0) & (np.abs(area) >= 1e-12))
    # One entry per triangle row, spanning the pixels the row could cover.
    ny = y1[drawn] - y0[drawn]
    t = np.repeat(drawn, ny)
    iy = y0[t] + np.arange(len(t)) - np.repeat(np.cumsum(ny) - ny, ny)
    lo, hi = row_span(x[t], y[t], area[t], iy + 0.5)
    lo, hi = np.clip(lo, x0[t] - 1, x1[t] + 1), np.clip(hi, x0[t] - 1, x1[t] + 1)
    start = np.maximum(np.ceil(lo - 0.5).astype(np.int64) - 1, x0[t])
    stop = np.minimum(np.floor(hi - 0.5).astype(np.int64) + 2, x1[t])
    rows = np.flatnonzero(stop > start)
    t, iy, start, counts = t[rows], iy[rows], start[rows], (stop - start)[rows]
    ends = np.cumsum(counts)
    first = 0
    while first < len(t):
        last = max(int(np.searchsorted(ends, ends[first] - counts[first] + FRAGMENTS)), first + 1)
        part = slice(first, last)
        fill_rows(img, zbuf, (x, y, z), colour, area, t[part], iy[part], start[part], counts[part])
        first = last


def row_span(x: Array, y: Array, area: Array, gy: Array) -> tuple[Array, Array]:
    """Per triangle row at pixel-centre height gy, the x range inside the triangle,
    solved from its three edge functions; fill_rows makes the exact test."""
    xa, xb, xc = x[:, 0], x[:, 1], x[:, 2]
    ya, yb, yc = y[:, 0], y[:, 1], y[:, 2]
    # Each edge function as slope * gx + offset, signed so inside is >= 0.
    s0 = -(yb - ya) / area
    o0 = ((xb - xa) * (gy - ya) + xa * (yb - ya)) / area
    s1 = (yc - ya) / area
    o1 = (-xa * (yc - ya) - (xc - xa) * (gy - ya)) / area
    s2, o2 = -(s0 + s1), 1 - (o0 + o1)
    lo = np.full(len(gy), -np.inf)
    hi = np.full(len(gy), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for slope, offset in ((s0, o0), (s1, o1), (s2, o2)):
            root = -offset / slope
            lo = np.where(slope > 0, np.maximum(lo, root), lo)
            hi = np.where(slope < 0, np.minimum(hi, root), hi)
            hi = np.where((slope == 0) & (offset < 0), -np.inf, hi)
    return lo, hi


def fill_rows(
    img: Array,
    zbuf: Array,
    corners: tuple[Array, Array, Array],
    colour: Array,
    area: Array,
    t: Ints,
    iy: Ints,
    start: Ints,
    counts: Ints,
) -> None:
    """Fill a run of triangle rows at once, row r covering `counts[r]` pixels from `start[r]`."""
    x, y, z = corners
    w = zbuf.shape[1]
    row = np.repeat(np.arange(len(t)), counts)
    ix = start[row] + np.arange(len(row)) - np.repeat(np.cumsum(counts) - counts, counts)
    t, iy = t[row], iy[row]
    gx, gy = ix + 0.5, iy + 0.5
    xa, xb, xc = x[t, 0], x[t, 1], x[t, 2]
    ya, yb, yc = y[t, 0], y[t, 1], y[t, 2]
    w0 = ((xb - xa) * (gy - ya) - (gx - xa) * (yb - ya)) / area[t]
    w1 = ((gx - xa) * (yc - ya) - (xc - xa) * (gy - ya)) / area[t]
    zi = z[t, 0] + w1 * (z[t, 1] - z[t, 0]) + w0 * (z[t, 2] - z[t, 0])
    pixel = iy * w + ix
    hit = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1) & (zi < zbuf.ravel()[pixel])
    pixel, zi, t = pixel[hit], zi[hit], t[hit]
    if not len(pixel):
        return
    # Nearest first, then the earliest drawn among equals, as one triangle at a time would keep.
    order = np.lexsort((t, zi, pixel))
    keep = order[np.r_[True, pixel[order][1:] != pixel[order][:-1]]]
    zbuf.ravel()[pixel[keep]] = zi[keep]
    img.reshape(-1, 3)[pixel[keep]] = colour[t[keep]]


def edge_points(edge: Shape[Any]) -> Array:
    """Points along an edge, no farther than TESS_TOL from it."""
    sampler = GCPnts_QuasiUniformDeflection(BRepAdaptor_Curve(edge.wrapped), TESS_TOL)
    if not sampler.IsDone():
        return np.empty((0, 3))
    points = (sampler.Value(i) for i in range(1, sampler.NbPoints() + 1))
    return np.array([(p.X(), p.Y(), p.Z()) for p in points])


Polylines = tuple[Array, Ints, npt.NDArray[np.bool_]]


def edge_polylines(shape: Shape[Any], flagged: set[int]) -> Polylines:
    """Every B-rep edge sampled once for all views: the points, the edge each point
    belongs to, and per edge whether it bounds an overhanging face."""
    points, owner, overhang = [], [], []
    for i, edge in enumerate(shape.edges()):
        p = edge_points(edge)
        points.append(p)
        owner.append(np.full(len(p), i))
        overhang.append(hash(edge) in flagged)
    if not points:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    return np.vstack(points), np.concatenate(owner), np.array(overhang)


def draw_edges(img: Array, zbuf: Array, lines: Polylines, to_px: Projector) -> None:
    """The B-rep edges, depth-tested; an overhanging face's edges are drawn wide and orange.
    Where edges cross, the pixel takes the colour of the later edge."""
    h, w = zbuf.shape
    finite = zbuf[np.isfinite(zbuf)]
    bias = max(np.ptp(finite) if finite.size else 1.0, 1.0) * 1e-3
    points, owner, overhang = lines
    ex, ey, ez = to_px(points)
    starts = np.flatnonzero(owner[:-1] == owner[1:])
    sx, sy, sz, segment = edge_samples(ex, ey, ez, starts)
    edge = owner[starts][segment]
    keep = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    sx, sy, sz, edge = sx[keep], sy[keep], sz[keep], edge[keep]
    seen = sz <= zbuf[sy, sx] + bias
    vx, vy, edge = sx[seen], sy[seen], edge[seen]
    wide = overhang[edge]
    xs, ys, edges = [vx], [vy], [edge]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        xs.append(np.clip(vx[wide] + dx, 0, w - 1))
        ys.append(np.clip(vy[wide] + dy, 0, h - 1))
        edges.append(edge[wide])
    pixel = np.concatenate(ys) * w + np.concatenate(xs)
    last = np.full(h * w, -1)
    np.maximum.at(last, pixel, np.concatenate(edges))
    hit = np.flatnonzero(last >= 0)
    img.reshape(-1, 3)[hit] = np.where(overhang[last[hit]][:, None], UNSUPPORTED, INK)


def edge_samples(ex: Array, ey: Array, ez: Array, starts: Ints) -> tuple[Ints, Ints, Array, Ints]:
    """Pixels along each segment from point `starts[k]` to the next, at least one per
    pixel step, sampled at the fractions np.linspace(0, 1, n + 1) gives; and each
    sample's segment."""
    after = starts + 1
    dx, dy, dz = ex[after] - ex[starts], ey[after] - ey[starts], ez[after] - ez[starts]
    n = (np.maximum(np.abs(dx), np.abs(dy))).astype(np.int64) + 1
    segment = np.repeat(np.arange(len(n)), n + 1)
    i = np.arange(len(segment)) - np.repeat(np.cumsum(n + 1) - (n + 1), n + 1)
    t = i * (1.0 / n)[segment]
    t[i == n[segment]] = 1.0
    at = starts[segment]
    sx = (ex[at] + t * dx[segment]).astype(int)
    sy = (ey[at] + t * dy[segment]).astype(int)
    return sx, sy, ez[at] + t * dz[segment], segment


def arrow(draw: ImageDraw.ImageDraw, tip: Array, along: Array) -> None:
    back = tip - along * ARROW_L
    perp = np.array([-along[1], along[0]]) * ARROW_W
    draw.polygon([tuple(tip), tuple(back + perp), tuple(back - perp)], fill=DIM)


def text_tile(text: str, font: Font, rotate: bool) -> Image.Image:
    box = font.getbbox(text)
    tile = Image.new("RGBA", (int(box[2] - box[0]) + 2, int(box[3] - box[1]) + 2), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((1 - box[0], 1 - box[1]), text, font=font, fill=(*DIM, 255))
    return tile.rotate(90, expand=True) if rotate else tile


def extent(tile: Image.Image, at: Array, out: Array) -> tuple[int, int, int, int]:
    """Where `tile` lands centred on `at`, pushed clear along `out`."""
    span = tile.height if abs(out[1]) > abs(out[0]) else tile.width
    c = at + out * (span / 2.0 + GAP)
    x, y = int(c[0] - tile.width / 2), int(c[1] - tile.height / 2)
    return x, y, x + tile.width, y + tile.height


def overlaps(a: Box2, b: Box2) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def fits(box: Box2, size: tuple[int, int]) -> bool:
    return box[0] >= GAP and box[1] >= GAP and box[2] <= size[0] - GAP and box[3] <= size[1] - GAP


def clear_of(box: Box2, blocker: Box2, out: Array) -> float:
    """Extra push along `out` that moves `box` off `blocker`, or 0 if it is clear."""
    if not overlaps(box, blocker):
        return 0.0
    i = 0 if abs(out[0]) > abs(out[1]) else 1
    span = (blocker[i + 2] - box[i]) if out[i] > 0 else (box[i + 2] - blocker[i])
    return span / max(abs(out[i]), 1e-6) + GAP


class Placement:
    """A dimension line pushed out by `offset` from the points p0, p1, laid out before drawing."""

    def __init__(
        self,
        size: tuple[int, int],
        p0: Any,
        p1: Any,
        offset: Any,
        text: str,
        font: Font,
    ) -> None:
        self.p0, self.p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
        offset = np.asarray(offset, dtype=float)
        self.q0, self.q1 = self.p0 + offset, self.p1 + offset
        self.out = offset / max(np.linalg.norm(offset), 1e-9)
        span = np.linalg.norm(self.q1 - self.q0)
        self.along = (self.q1 - self.q0) / max(span, 1e-9)
        self.tile = text_tile(text, font, abs(self.along[1]) > abs(self.along[0]))
        if span < 4 * ARROW_L:
            at, push = self.q1 + self.along * ARROW_L, self.along
            x0, y0, x1, y1 = extent(self.tile, at, push)
            if x0 < 0 or y0 < 0 or x1 > size[0] or y1 > size[1]:
                at, push = self.q0 - self.along * ARROW_L, -self.along
        else:
            at, push = (self.q0 + self.q1) / 2.0, self.out
        box = extent(self.tile, at, push)
        self.text_at = (box[0], box[1])
        self.text_box: Box2 = box
        self.box: Box2 = (
            min(box[0], self.q0[0], self.q1[0]),
            min(box[1], self.q0[1], self.q1[1]),
            max(box[2], self.q0[0], self.q1[0]),
            max(box[3], self.q0[1], self.q1[1]),
        )

    def draw(self, im: Image.Image, draw: ImageDraw.ImageDraw) -> None:
        for p, q in ((self.p0, self.q0), (self.p1, self.q1)):
            draw.line([tuple(p + self.out * GAP), tuple(q + self.out * OVER)], fill=DIM, width=LW)
        draw.line([tuple(self.q0), tuple(self.q1)], fill=DIM, width=LW)
        arrow(draw, self.q0, self.along)
        arrow(draw, self.q1, -self.along)
        im.paste(self.tile, self.text_at, self.tile)


def leader(
    im: Image.Image,
    draw: ImageDraw.ImageDraw,
    p0: Array,
    p1: Array,
    centre: Array,
    bounds: Box2,
    text: str,
    font: Font,
) -> Box2:
    """A dimension drawn on the feature, its text led out past `bounds`; returns the text box.
    For a radial feature no perpendicular push reaches a margin without crossing the part."""
    along = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
    draw.line([tuple(p0), tuple(p1)], fill=DIM, width=LW)
    arrow(draw, p0, along)
    arrow(draw, p1, -along)
    mid = (p0 + p1) / 2.0
    out = mid - centre
    n = np.linalg.norm(out)
    out = along if n < 1.0 else out / n
    x0, y0, x1, y1 = bounds
    reach = min(
        (((x1 if out[i] > 0 else x0) if i == 0 else (y1 if out[i] > 0 else y0)) - mid[i]) / out[i]
        for i in (0, 1)
        if abs(out[i]) > 1e-6
    )
    tip = mid + out * (max(reach, 0.0) + OFF)
    tile = text_tile(text, font, abs(out[1]) > abs(out[0]))
    box = extent(tile, tip, out)
    shift = np.array(
        [
            max(0, OFF - box[0]) - max(0, box[2] - (im.width - OFF)),
            max(0, OFF - box[1]) - max(0, box[3] - (im.height - OFF)),
        ]
    )
    tip = tip + shift
    x, y = int(box[0] + shift[0]), int(box[1] + shift[1])
    draw.line([tuple(mid + out * GAP), tuple(tip)], fill=DIM, width=LW)
    im.paste(tile, (x, y), tile)
    return x, y, x + tile.width, y + tile.height


def annotate(
    im: Image.Image,
    name: str,
    detail: str | None,
    shape: Shape[Any],
    dims: Sequence[Dimension],
    right: Array,
    up: Array,
    fwd: Array,
    to_px: Projector,
    shaded: bool,
) -> None:
    """Name the view and dimension it: features first, then overall size, then leaders."""
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=FONT_PX)
    draw.text((GAP, GAP), f"detail: {name}" if detail else name, font=font, fill=DIM)
    bb = shape.bounding_box()
    lengths = (bb.size.X, bb.size.Y, bb.size.Z)
    ra, ua = axis_of(right), axis_of(up)
    if ra is None or ua is None:
        draw.text(
            (GAP, GAP + FONT_PX + GAP),
            "{:.1f} x {:.1f} x {:.1f} mm".format(*lengths),
            font=font,
            fill=DIM,
        )
        if shaded:
            note = "orange faces print unsupported (over 45 deg)"
            draw.text((GAP, GAP + 2 * (FONT_PX + GAP)), note, font=font, fill=UNSUPPORTED)
        return

    mine = [
        d
        for d in dims
        if (d.get("detail") == detail if detail else not d.get("detail") and d["view"] == name)
    ]
    corners = list(itertools.product(*zip(tuple(bb.min), tuple(bb.max), strict=True)))
    if detail:
        corners = [p for d in mine for p in (d["a"], d["b"])]
    px, py, _ = to_px(corners)
    x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
    centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    size = (im.width, im.height)

    taken: list[Box2] = []
    texts: list[Box2] = []
    led: list[tuple[Array, Array, str]] = []
    for dim in mine:
        a, b = np.array(dim["a"], dtype=float), np.array(dim["b"], dtype=float)
        length = float(np.linalg.norm(b - a))
        if abs((b - a) @ fwd) > 1e-6 * max(length, 1.0):
            print(f"  skip '{dim['label']}': foreshortened in the {name} view")
            continue
        (ax, ay, _), (bx, by, _) = to_px([a]), to_px([b])
        pa, pb = np.array([ax[0], ay[0]]), np.array([bx[0], by[0]])
        v = pb - pa
        n = float(np.linalg.norm(v))
        if n < 1.0:
            continue
        out = np.array([-v[1], v[0]]) / n
        mid = (pa + pb) / 2.0
        side = out @ (mid - centre)
        if side < -1.0 or (abs(side) <= 1.0 and out[0] + out[1] < 0):
            out = -out
        reach = abs(out[0]) * (x1 - mid[0] if out[0] > 0 else mid[0] - x0) + abs(out[1]) * (
            y1 - mid[1] if out[1] > 0 else mid[1] - y0
        )
        label = f"{dim['label']} {length:.2f}" if detail else f"{dim['label']} {length:.1f}"
        push = max(reach, 0.0) + FEAT_OFF
        for _ in range(len(taken) + 1):
            plan = Placement(size, pa, pb, out * push, label, font)
            extra = max([clear_of(plan.box, t, out) for t in taken] + [0.0])
            if extra <= 0.0:
                break
            push += extra
        axis_aligned = min(abs(v[0]), abs(v[1])) / n < 0.035
        blocked = any(clear_of(plan.box, t, out) > 0.0 for t in taken)
        if not fits(plan.box, size) or (not axis_aligned and reach > n) or blocked:
            led.append((pa, pb, label))
        else:
            plan.draw(im, draw)
            taken.append(plan.box)
            texts.append(plan.text_box)

    edge: list[Box2] = []
    if not detail:
        below = max([t[3] for t in taken] + [y1]) + OFF
        beside = max([t[2] for t in taken] + [x1]) + OFF
        for p0, p1, offset, value in (
            ((x0, y1), (x1, y1), (0, below - y1), lengths[ra]),
            ((x1, y0), (x1, y1), (beside - x1, 0), lengths[ua]),
        ):
            plan = Placement(size, p0, p1, offset, f"{value:.1f}", font)
            plan.draw(im, draw)
            edge.append(plan.box)
            texts.append(plan.text_box)
    placed = taken + edge
    clear = (
        min([t[0] for t in placed] + [x0]),
        min([t[1] for t in placed] + [y0]),
        max([t[2] for t in placed] + [x1]),
        max([t[3] for t in placed] + [y1]),
    )
    texts += [leader(im, draw, pa, pb, centre, clear, label, font) for pa, pb, label in led]
    for one, other in itertools.combinations(texts, 2):
        if overlaps(one, other):
            raise ValueError(f"{name}: labels overlap, {one} and {other}")


def render_view(
    shape: Shape[Any],
    groups: list[Group],
    lines: Polylines,
    view: tuple[str, Any, Any, str | None],
    size: tuple[int, int],
    scale: float,
    centre: Array,
    dims: Sequence[Dimension],
) -> Image.Image:
    name, direction, up_hint, detail = view
    w, h = size
    right, up, fwd = basis(direction, up_hint)
    to_px = projector(right, up, fwd, size, scale, centre)
    img = np.empty((h, w, 3))
    img[:] = BG
    zbuf = np.full((h, w), np.inf)
    rasterise(img, zbuf, groups, to_px, unit(-fwd + 0.45 * up + 0.35 * right))
    draw_edges(img, zbuf, lines, to_px)
    tile = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    shaded = any(u.any() for _, _, u in groups)
    annotate(tile, name, detail, shape, dims, right, up, fwd, to_px, shaded)
    return tile


def detail_frames(
    shape: Shape[Any],
    dims: Sequence[Dimension],
    size: tuple[int, int],
    build_up: Sequence[float] | None,
) -> list[tuple[Any, ...]]:
    """One zoomed, sectioned view per named detail, framed on what its dimensions span."""
    frames = []
    for name in sorted({d["detail"] for d in dims if d.get("detail")}):
        group = [d for d in dims if d.get("detail") == name]
        views = sorted({d["view"] for d in group})
        if len(views) != 1:
            raise ValueError(f"detail '{name}' spans views {views}")
        direction, up_hint = next((d, u) for n, d, u in VIEWS if n == views[0])
        right, up, fwd = basis(direction, up_hint)
        pts = np.array([d[k] for d in group for k in ("a", "b")], dtype=float)
        xy = np.column_stack([pts @ right, pts @ up])
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        pad = (hi - lo) * (DETAIL_PAD - 1) / 2
        centre, scale = fit(lo - pad, hi + pad, size)
        cut = section(shape, fwd, float(np.mean(pts @ fwd)))
        cut_groups, cut_flagged = tessellate(cut, build_up)
        frames.append(
            ((name, direction, up_hint, name), centre, scale, cut, cut_groups, cut_flagged)
        )
    return frames


def draw_sheet(
    shape: Shape[Any],
    groups: list[Group],
    flagged: set[int],
    dims: Sequence[Dimension],
    build_up: Sequence[float] | None,
    path: Path,
) -> None:
    """Six views at one shared scale, plus a tile per detail, to one PNG."""
    size = (TILE[0] * SS, TILE[1] * SS)
    frames = [view_frame(groups, d, u, size) for _, d, u in VIEWS]
    scale = min(s for _, s in frames)
    plan = [
        ((n, d, u, None), c, scale, shape, groups, flagged)
        for (n, d, u), (c, _) in zip(VIEWS, frames, strict=True)
    ]
    plan += detail_frames(shape, dims, size, build_up)
    rows = -(-len(plan) // COLS)
    sheet = Image.new("RGB", (size[0] * COLS, size[1] * rows), BG)
    lines: dict[int, Polylines] = {}
    for _, _, _, seen, _, flags in plan:
        if id(seen) not in lines:
            lines[id(seen)] = edge_polylines(seen, flags)
    for i, (view, centre, s, seen, tris, _) in enumerate(plan):
        tile = render_view(seen, tris, lines[id(seen)], view, size, s, centre, dims)
        sheet.paste(tile, ((i % COLS) * size[0], (i // COLS) * size[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.resize((size[0] * COLS // SS, size[1] * rows // SS), Image.Resampling.LANCZOS).save(path)
    print(f"wrote {path}")


def draw_part(stem: str, shape: Shape[Any], spec: Mapping[str, Any], out_dir: Path) -> None:
    build_up = spec.get("build_up")
    if build_up is None:
        print(f"  {stem}: no build direction declared, overhangs not shaded")
    groups, flagged = tessellate(shape, build_up)
    draw_sheet(
        shape, groups, flagged, spec.get("dimensions", []), build_up, out_dir / f"{stem}.png"
    )


def draw_assembly(
    name: str, spec: Mapping[str, Any], seat: Callable[[str], Shape[Any]], out_dir: Path
) -> None:
    """Every seated part in its assembled place, each with its own build direction."""
    shapes: list[Shape[Any]] = []
    groups: list[Group] = []
    flagged: set[int] = set()
    for part in spec["seat"]:
        shape = Pos(*part.get("at", (0.0, 0.0, 0.0))) * seat(part["stem"])
        g, f = tessellate(shape, part.get("build_up"))
        shapes.append(shape)
        groups += g
        flagged |= f
    draw_sheet(
        Compound(shapes), groups, flagged, spec.get("dimensions", []), None, out_dir / f"{name}.png"
    )


def draw_sheets(
    shapes: Mapping[str, Shape[Any]],
    spec: Mapping[str, Any],
    seat: Callable[[str], Shape[Any]],
    out_dir: Path,
) -> None:
    """A sheet per shape, keyed by file stem, and one per assembly in `spec`.
    `seat` returns the shape for an assembly's stem, which may be ../<model>/<stem>."""
    parts = spec.get("parts", {})
    for stem, shape in shapes.items():
        draw_part(stem, shape, parts.get(stem, {}), out_dir)
    for name, assembly in sorted(spec.get("assemblies", {}).items()):
        draw_assembly(name, assembly, seat, out_dir)
