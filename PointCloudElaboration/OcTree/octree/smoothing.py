"""Surface smoothing: flatten stepped voxels into planar OpenStudio surfaces.

OpenStudio / EnergyPlus need planar, well-formed surfaces; the raw voxel wall
is stepped. `smooth_surface` finds the surface's plane, projects the voxels
onto it, and keeps each voxel's semantic class so the flattened wall is
subdivided into homogeneous sub-surfaces (wall, window, door, ...).

Finding the plane (default `offset_method="ransac"`):
- A RANSAC/MSAC dominant-plane fit (pure numpy, `fit_plane_ransac` /
  `extract_planes`) locates the actual best-fit plane of the voxels — any
  orientation, not just world x/y/z. This replaces the older heuristic of
  "pick the mode voxel layer along a fixed axis", which cut across a facade
  whenever the wall was not aligned with the grid or was slightly tilted.
  The `u`/`v`/`z` selector now chooses *which* detected plane to flatten:
  `u` = the dominant (largest) vertical facade, `v` = the perpendicular
  facade, `z` = the dominant horizontal plane (roof / floor). `x`/`y` pick
  the plane whose normal is closest to that world axis.
- The plane's local 2-D basis (e_u, e_v) comes from PCA on its inliers
  (`plane_basis`), so u runs along the wall and v across it.

Legacy `offset_method` values are kept for comparison:
- `mode` (most-populated voxel layer) | `median` | `outer` (95th-pct face),
  flattened along a literal world axis `x/y/z`, or the PCA-yaw-aligned `u/v`
  (`principal_yaw` corrects only rotation about vertical z).

Snapping to the plane (both methods):
- A tolerance band (+/- `tolerance_voxels`) decides membership: voxels within
  the band snap onto the plane (a recessed window is only 1-3 voxels deep, so
  it snaps flush -> a co-planar sub-surface, which is what OpenStudio requires
  for fenestration); voxels beyond the band are returned as `deviations`
  (kept, not dropped) so the caller can treat them as noise or their own
  surface.

Pure numpy on purpose (no sklearn / open3d / scipy): the RANSAC fit, the SVD
refit and the PCA basis are all implemented here so smoothing runs headless.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classes import MAX_CLASS_ID, UNKNOWN_CLASS_ID, class_name
from .voxelizer import VoxelGrid

_AXES = {"x": 0, "y": 1, "z": 2}

# Semantic classes treated as building envelope (base Surface) vs fenestration
# (SubSurface) when exporting to OpenStudio. Ids follow classes.py.
_FENESTRATION_IDS = {2, 3, 14}  # window, door, blinds
_ENVELOPE_ROLE = "envelope"
_FENESTRATION_ROLE = "fenestration"

# A plane's normal is "vertical" (facade) when its z-component is small, and
# "horizontal" (roof / floor) when its z-component is large. Used to route the
# u/v (facade) and z (roof/floor) selectors to the right RANSAC plane.
_VERTICAL_NZ_MAX = 0.35
_HORIZONTAL_NZ_MIN = 0.70

# Cap the number of voxels the RANSAC hypothesis search runs on: a few thousand
# random voxels are plenty to lock onto the dominant plane, and it keeps the
# interactive smoothing snappy. The chosen plane is then applied to *all* voxels.
_RANSAC_FIT_CAP = 20_000


@dataclass
class SubSurface:
    class_id: int
    class_name: str
    role: str  # "envelope" | "fenestration"
    polygons: list[np.ndarray]  # each (4, 3) planar quad, world coords, CCW about +normal


@dataclass
class PlanarSurface:
    axis: str  # 'x' | 'y' | 'z' | 'u' | 'v' — the surface-normal axis (flattened)
    plane_coord: float  # signed offset of the plane along its normal (world);
    #                     origin . normal for the RANSAC fit, the world axis
    #                     coordinate for the legacy literal-axis flatten
    voxel_size: float
    subsurfaces: list[SubSurface] = field(default_factory=list)
    n_inliers: int = 0
    n_deviations: int = 0
    deviations: VoxelGrid | None = None  # voxels beyond the tolerance band, world coords
    rotation_deg: float = 0.0  # in-plane yaw of the fitted/aligned wall, degrees
    normal: np.ndarray | None = None  # (3,) unit plane normal (RANSAC fit); None for legacy
    origin: np.ndarray | None = None  # (3,) a point on the plane (RANSAC fit); None for legacy
    e_u: np.ndarray | None = None  # (3,) in-plane basis axis (RANSAC fit); None for legacy
    e_v: np.ndarray | None = None  # (3,) in-plane basis axis (RANSAC fit); None for legacy
    n_filled: int = 0  # enclosed voids filled from agreeing neighbours
    n_unknown: int = 0  # enclosed voids filled as the "unknown" material

    @property
    def n_polygons(self) -> int:
        return sum(len(s.polygons) for s in self.subsurfaces)


# --------------------------------------------------------------------------- #
# RANSAC / MSAC dominant-plane fit (pure numpy) — finds the plane to flatten   #
# onto, replacing the old "mode voxel layer along a fixed axis" heuristic.     #
# --------------------------------------------------------------------------- #
@dataclass
class Plane:
    normal: np.ndarray  # (3,) unit normal
    point: np.ndarray  # (3,) a point on the plane (inlier centroid after refit)
    inliers: np.ndarray  # (N,) bool over the fitted points


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float,
    iters: int = 500,
    seed: int = 0,
    cost: str = "msac",
) -> Plane:
    """Fit the dominant plane by RANSAC (inlier count) or MSAC (truncated L2).

    `threshold` is the inlier distance in metres. MSAC scores each hypothesis
    by sum(min(dist^2, threshold^2)) (lower is better) — it rewards tight
    inliers, unlike plain RANSAC which only counts them. After the best
    hypothesis, the normal is refit to all its inliers via SVD (total-least-
    squares), then inliers are recomputed against the refined plane.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        raise ValueError("need at least 3 points to fit a plane")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    rng = np.random.default_rng(seed)
    t2 = threshold * threshold
    best_score = np.inf
    best_inliers = None

    for _ in range(int(iters)):
        i = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[i]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue  # collinear sample, skip
        nrm = nrm / norm
        dist = np.abs((pts - p0) @ nrm)
        inl = dist < threshold
        if cost == "msac":
            score = float(np.minimum(dist * dist, t2).sum())  # minimize
        elif cost == "ransac":
            score = -float(inl.sum())  # minimize negative inlier count
        else:
            raise ValueError(f"cost must be 'msac' or 'ransac' (got {cost!r})")
        if score < best_score:
            best_score = score
            best_inliers = inl

    if best_inliers is None or best_inliers.sum() < 3:
        raise RuntimeError("RANSAC found no plane (try a larger threshold/iters)")

    # Refit the normal to all inliers (total least squares via SVD).
    inl_pts = pts[best_inliers]
    centroid = inl_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inl_pts - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])  # smallest-variance direction

    dist = np.abs((pts - centroid) @ normal)
    inliers = dist < threshold
    return Plane(normal=normal, point=centroid, inliers=inliers)


def extract_planes(
    points: np.ndarray,
    threshold: float,
    iters: int = 500,
    seed: int = 0,
    cost: str = "msac",
    max_planes: int = 5,
    min_inliers: int = 100,
) -> list[Plane]:
    """Iteratively RANSAC-fit planes: fit, strip inliers, repeat.

    Returns up to `max_planes` planes, each with an `inliers` mask over the
    *original* `points`, sorted by inlier count (descending). Stops early when
    the next plane would have fewer than `min_inliers`. This gives the caller a
    *choice* of wall (dominant facade / perpendicular facade / roof) instead of
    only the single most-populated plane.
    """
    pts = np.asarray(points, dtype=np.float64)
    widx = np.arange(len(pts))  # indices of the still-unassigned points
    work = pts
    planes: list[Plane] = []
    for k in range(max_planes):
        if len(work) < max(3, min_inliers):
            break
        pl = fit_plane_ransac(work, threshold, iters, seed + k, cost)
        if int(pl.inliers.sum()) < min_inliers:
            break
        full = np.zeros(len(pts), dtype=bool)
        full[widx[pl.inliers]] = True
        planes.append(Plane(normal=pl.normal, point=pl.point, inliers=full))
        keep = ~pl.inliers
        work, widx = work[keep], widx[keep]
    planes.sort(key=lambda p: -int(p.inliers.sum()))
    return planes


def _any_perpendicular(n: np.ndarray) -> np.ndarray:
    """Some unit vector orthogonal to n (numerically safe seed axis)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e = a - (a @ n) * n
    return e / np.linalg.norm(e)


def plane_basis(normal: np.ndarray, inlier_points: np.ndarray):
    """Orthonormal in-plane axes (e_u, e_v) aligned with the wall, plus origin.

    PCA on the inliers' in-plane spread picks e_u along the dominant wall
    direction (for a vertical wall that is the wide horizontal direction), and
    e_v = normal x e_u completes a right-handed frame so a CCW winding in (u, v)
    is CCW about +normal. Returns (origin, e_u, e_v, n) with origin the inlier
    centroid (a point on the plane). Mirrors principal_yaw, in 3-D.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    pts = np.asarray(inlier_points, dtype=np.float64)
    origin = pts.mean(axis=0)

    centered = pts - origin
    in_plane = centered - np.outer(centered @ n, n)  # drop the normal component
    cov = in_plane.T @ in_plane
    evals, evecs = np.linalg.eigh(cov)
    e_u = evecs[:, int(np.argmax(evals))]
    e_u = e_u - (e_u @ n) * n
    norm = np.linalg.norm(e_u)
    e_u = e_u / norm if norm > 1e-9 else _any_perpendicular(n)
    e_v = np.cross(n, e_u)
    e_v = e_v / np.linalg.norm(e_v)
    return origin, e_u, e_v, n


def project_to_plane(pts, origin, e_u, e_v, normal):
    """In-plane (u, v) coordinates and signed perpendicular offset d of pts."""
    d = np.asarray(pts, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    return d @ e_u, d @ e_v, d @ normal


def _axis_target(axis: str, yaw_deg: float) -> np.ndarray:
    """Stable unit target normal the axis selector locks onto.

    `u`/`v` come from the building's PCA yaw (a perpendicular horizontal pair),
    so they stay on the *same two facades* regardless of voxel size; `z` is
    vertical; `x`/`y` are the literal world axes. `principal_yaw` is stable
    across voxel sizes, which is what anchors the selection.
    """
    th = np.radians(yaw_deg)
    if axis == "u":
        return np.array([np.cos(th), np.sin(th), 0.0])   # dominant facade
    if axis == "v":
        return np.array([-np.sin(th), np.cos(th), 0.0])  # perpendicular facade
    if axis == "z":
        return np.array([0.0, 0.0, 1.0])
    if axis == "x":
        return np.array([1.0, 0.0, 0.0])
    if axis == "y":
        return np.array([0.0, 1.0, 0.0])
    raise ValueError(f"axis must be one of u, v, z, x, y (got {axis!r})")


def _select_plane_for_axis(planes: list[Plane], axis: str, target: np.ndarray) -> Plane | None:
    """Pick the plane whose normal best matches a *stable* target direction.

    Within the orientation-appropriate pool, choose the plane maximizing
    `|n . target|` — not the largest plane of a class. Two near-equal
    perpendicular facades would otherwise swap which one is "largest" as the
    voxel size changes, flipping the surface for a fixed axis; anchoring to the
    PCA-derived `target` (see `_axis_target`) keeps each axis locked to the same
    physical wall. Returns `None` for `z` when no horizontal plane exists (a
    facade scan at fine resolution) so a wall is never mislabeled roof/floor.
    `planes` is sorted largest-first, so ties break toward the bigger plane.
    """
    if not planes:
        return None
    if axis in ("u", "v"):
        pool = [p for p in planes if abs(p.normal[2]) < _VERTICAL_NZ_MAX] or planes
    elif axis == "z":
        pool = [p for p in planes if abs(p.normal[2]) > _HORIZONTAL_NZ_MIN]
        if not pool:
            return None  # no roof/floor at this resolution — don't fake it
    else:  # literal 'x' / 'y'
        pool = planes
    return max(pool, key=lambda p: abs(float(p.normal @ target)))


# --------------------------------------------------------------------------- #
# Legacy literal-axis / PCA-yaw helpers (offset_method mode|median|outer)      #
# --------------------------------------------------------------------------- #
def _layer_indices(grid: VoxelGrid) -> np.ndarray:
    """Integer voxel-lattice index (i, j, k) of each voxel center."""
    return np.round((grid.centers - grid.origin) / grid.voxel_size - 0.5).astype(np.int64)


def _plane_layer(layers_along_axis: np.ndarray, method: str) -> int:
    col = layers_along_axis
    if method == "mode":
        vals, counts = np.unique(col, return_counts=True)
        return int(vals[counts.argmax()])
    if method == "median":
        return int(round(float(np.median(col))))
    if method == "outer":
        return int(round(float(np.percentile(col, 95))))
    raise ValueError(f"Unknown offset_method '{method}' (ransac | mode | median | outer)")


def _role(class_id: int) -> str:
    return _FENESTRATION_ROLE if class_id in _FENESTRATION_IDS else _ENVELOPE_ROLE


def principal_yaw(grid: VoxelGrid, select: np.ndarray | None = None) -> float:
    """Dominant horizontal direction of the voxels, in degrees (0-180).

    PCA on the (x, y) voxel centers: the first principal component is the
    building's actual wall direction, which real scans are rarely aligned
    with. Stable across which classes are included (verified: wall-only,
    envelope-only and all-classes all agree to within ~1 degree on the sample
    building), so the default is to use whatever grid/selection is passed in.
    Returns an angle mod 180 deg since a direction and its 180-deg opposite
    describe the same wall orientation.
    """
    mask = np.ones(len(grid), bool) if select is None else np.asarray(select, bool)
    xy = grid.centers[mask][:, :2]
    xy = xy - xy.mean(axis=0)
    cov = np.cov(xy.T)
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, np.argmax(evals)]
    return float(np.degrees(np.arctan2(principal[1], principal[0])) % 180.0)


def _rotate_grid_horizontal(grid: VoxelGrid, yaw_deg: float):
    """Rotate `grid`'s centers by -yaw_deg about their horizontal centroid.

    Keeps z unchanged (yaw only). Returns (rotated_grid, pivot_xy) — pivot_xy
    is needed to rotate the output back into world coordinates later.

    voxelize() uses a *corner* origin: cell centers sit half a voxel above it
    (centers.min() == origin + 0.5*voxel_size), and _layer_indices/_plane_layer
    rely on that offset. The rotated centers have no corners of their own (an
    arbitrary rotation doesn't preserve the axis-aligned lattice), so origin is
    reconstructed with the same half-voxel convention: centers.min() - 0.5*vs.
    Using centers.min() directly here would shift every layer index by 0.5 and
    corrupt the mode/tolerance-band computation.
    """
    pivot = grid.centers[:, :2].mean(axis=0)
    theta = -np.radians(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    vs = grid.voxel_size

    xy = (grid.centers[:, :2] - pivot) @ rot.T
    centers = np.column_stack([xy, grid.centers[:, 2]])
    origin = np.array([xy[:, 0].min() - 0.5 * vs, xy[:, 1].min() - 0.5 * vs, grid.origin[2]])

    rotated = VoxelGrid(
        centers=centers, labels=grid.labels, counts=grid.counts,
        voxel_size=grid.voxel_size, origin=origin,
    )
    return rotated, pivot


def _rotate_xy_back(xy: np.ndarray, yaw_deg: float, pivot: np.ndarray) -> np.ndarray:
    """Inverse of _rotate_grid_horizontal's rotation, for (..., 2) xy arrays."""
    theta = np.radians(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return xy @ rot.T + pivot


def _rotate_surface_back(surface: "PlanarSurface", yaw_deg: float, pivot: np.ndarray) -> None:
    """Rotate a surface's polygons and deviations' xy back into world coords, in place."""
    for sub in surface.subsurfaces:
        sub.polygons = [
            np.column_stack([_rotate_xy_back(poly[:, [0, 1]], yaw_deg, pivot), poly[:, 2]])
            for poly in sub.polygons
        ]
    if surface.deviations is not None and len(surface.deviations):
        d = surface.deviations
        xy = _rotate_xy_back(d.centers[:, [0, 1]], yaw_deg, pivot)
        d.centers = np.column_stack([xy, d.centers[:, 2]])


# --------------------------------------------------------------------------- #
# Shared class-zoning helpers                                                 #
# --------------------------------------------------------------------------- #
def _greedy_rectangles(mask: np.ndarray):
    """Cover the True cells of a 2-D boolean mask with axis-aligned rectangles.

    Greedy: for each uncovered cell (row-major), grow right along the row, then
    down as long as the full-width strip stays True and uncovered. Yields
    (r0, c0, r1, c1) inclusive rectangles. Not minimal, but few and simple, and
    separate blobs (e.g. two windows) never merge because they are not adjacent.
    """
    remaining = mask.copy()
    n_rows, n_cols = mask.shape
    rects = []
    for r0 in range(n_rows):
        c = 0
        while c < n_cols:
            if not remaining[r0, c]:
                c += 1
                continue
            # grow right
            c1 = c
            while c1 + 1 < n_cols and remaining[r0, c1 + 1]:
                c1 += 1
            # grow down while the whole [c..c1] strip is available
            r1 = r0
            while r1 + 1 < n_rows and remaining[r1 + 1, c : c1 + 1].all():
                r1 += 1
            remaining[r0 : r1 + 1, c : c1 + 1] = False
            rects.append((r0, c, r1, c1))
            c = c1 + 1
    return rects


def _zone_by_class(cell_ij: np.ndarray, labels: np.ndarray):
    """Majority class per integer cell -> a dense raster (-1 empty) + its (i0, j0).

    `cell_ij` is (N, 2) integer cell indices and `labels` the per-voxel class.
    Returns (raster, i0, j0) where raster[i - i0, j - j0] is the majority class
    of that cell (reusing the histogram idea from voxelizer._grid_from_index).
    """
    keys, inverse = np.unique(cell_ij, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    hist = np.zeros((len(keys), MAX_CLASS_ID + 1), np.int64)
    np.add.at(hist, (inverse, np.clip(labels, 0, MAX_CLASS_ID)), 1)
    cell_labels = hist.argmax(axis=1)

    i0, j0 = keys[:, 0].min(), keys[:, 1].min()
    n_i = keys[:, 0].max() - i0 + 1
    n_j = keys[:, 1].max() - j0 + 1
    raster = np.full((n_i, n_j), -1, np.int64)
    raster[keys[:, 0] - i0, keys[:, 1] - j0] = cell_labels
    return raster, int(i0), int(j0)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels of a 2-D boolean mask (plain BFS flood fill).

    Returns (labels, n_components): `labels[i, j]` is the component index
    (>= 0) where `mask[i, j]` is True, and -1 elsewhere. Pure numpy/Python (no
    scipy.ndimage.label) so the pipeline stays dependency-free; fast enough for
    a wall raster (tens of thousands of cells).
    """
    from collections import deque

    n_rows, n_cols = mask.shape
    labels = np.full((n_rows, n_cols), -1, dtype=np.int64)
    n_components = 0
    for i0 in range(n_rows):
        for j0 in range(n_cols):
            if not mask[i0, j0] or labels[i0, j0] != -1:
                continue
            labels[i0, j0] = n_components
            queue = deque([(i0, j0)])
            while queue:
                r, c = queue.popleft()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < n_rows and 0 <= nc < n_cols \
                            and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = n_components
                        queue.append((nr, nc))
            n_components += 1
    return labels, n_components


def fill_enclosed_cells(
    raster: np.ndarray, unknown_id: int = UNKNOWN_CLASS_ID
) -> tuple[np.ndarray, int, int]:
    """Fill empty regions of any size that are fully enclosed by occupied cells.

    `raster` is a dense 2-D class grid (`-1` = empty), as built by
    `_zone_by_class`. Empty cells are grouped into 4-connected regions
    (`_label_components`); a region is "enclosed" when it never touches the
    raster border — i.e. every cell is occupied all the way around it, like a
    hole punched inside a wall or window rather than a notch open to the edge
    (a border-touching region, e.g. sparse terrain at the scan's edge, is left
    alone). Each enclosed region is filled with:
    - the single class found all around its border, when every bordering cell
      agrees (e.g. a multi-cell gap inside a window becomes window), or
    - `unknown_id` (a new "unknown" material), when the border has more than
      one class.

    The fill decision reads the *original* raster, so it does not depend on
    region processing order. Returns (filled_raster, n_same, n_unknown),
    counting *cells* filled — a 4-cell hole filled one class counts as 4.
    """
    raster = np.asarray(raster)
    filled = raster.copy()
    if raster.ndim != 2 or raster.size == 0:
        return filled, 0, 0

    empty = raster < 0
    if not empty.any():
        return filled, 0, 0
    labels, n_components = _label_components(empty)
    n_rows, n_cols = raster.shape

    n_same = n_unknown = 0
    for comp in range(n_components):
        cells = np.argwhere(labels == comp)
        rows, cols = cells[:, 0], cells[:, 1]
        if (rows == 0).any() or (rows == n_rows - 1).any() \
                or (cols == 0).any() or (cols == n_cols - 1).any():
            continue  # touches the raster border: open to the outside, not a hole

        border_classes = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                cls = raster[nr, nc]
                if cls >= 0:
                    border_classes.add(int(cls))

        if len(border_classes) == 1:
            fill_value = next(iter(border_classes))
            n_same += len(cells)
        else:
            fill_value = unknown_id
            n_unknown += len(cells)
        filled[rows, cols] = fill_value
    return filled, n_same, n_unknown


def _quad(u0, u1, v0, v1, plane_coord, axis_idx, u_idx, v_idx):
    """Planar quad (4,3) for in-plane cell-corner spans [u0,u1] x [v0,v1] (world axes)."""
    corners_uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]  # CCW in the u-v plane
    quad = np.empty((4, 3), dtype=float)
    for row, (u, v) in enumerate(corners_uv):
        quad[row, axis_idx] = plane_coord
        quad[row, u_idx] = u
        quad[row, v_idx] = v
    return quad


def _quad_uv(u0, u1, v0, v1, origin, e_u, e_v):
    """Planar 3-D quad (4,3) for the in-plane span [u0,u1] x [v0,v1] on a basis.

    Corners wound CCW about +normal (= e_u x e_v), matching _quad's winding, so
    the RANSAC and legacy paths produce consistently oriented polygons.
    """
    corners_uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    return np.array(
        [origin + uu * e_u + vv * e_v for uu, vv in corners_uv], dtype=float
    )


@dataclass
class PlaneAnchor:
    """A previously computed RANSAC plane, reusable to keep a surface put.

    Pass this as `smooth_surface(..., anchor=...)` (RANSAC path only) to skip
    the RANSAC fit and axis selection entirely and reproject onto this exact
    plane instead. Without an anchor, `smooth_surface` re-fits the plane from
    scratch every call, and a different voxel size can yield a slightly
    different plane (different inlier set -> different position/orientation);
    with an anchor, only the raster's resolution changes as voxel size changes
    — the plane itself stays fixed. Build one from a previous result via
    `PlaneAnchor.from_surface(surface)`.
    """

    normal: np.ndarray
    origin: np.ndarray
    e_u: np.ndarray
    e_v: np.ndarray

    @staticmethod
    def from_surface(surface: "PlanarSurface") -> "PlaneAnchor | None":
        """Capture the plane a RANSAC-fitted surface was built on, or None (legacy fit / no plane)."""
        if surface.normal is None or surface.origin is None:
            return None
        return PlaneAnchor(
            normal=surface.normal.copy(), origin=surface.origin.copy(),
            e_u=surface.e_u.copy(), e_v=surface.e_v.copy(),
        )


def smooth_surface(
    grid: VoxelGrid,
    axis: str,
    offset_method: str = "ransac",
    tolerance_voxels: int = 3,
    select: np.ndarray | None = None,
    rotation_deg: float | None = None,
    ransac_threshold: float | None = None,
    ransac_iters: int = 500,
    seed: int = 0,
    anchor: PlaneAnchor | None = None,
) -> PlanarSurface:
    """Flatten `grid` onto a single plane, preserving class zoning.

    offset_method:
        'ransac' (default) — find the plane with a RANSAC/MSAC dominant-plane
            fit (any orientation); `axis` selects which detected plane to use:
            'u' dominant facade, 'v' perpendicular facade, 'z' roof/floor,
            'x'/'y' nearest to that world axis. `ransac_threshold` (metres,
            default ~0.5*voxel_size) is the plane-fit inlier distance;
            `ransac_iters` / `seed` control the (deterministic) hypothesis search.
        'mode' | 'median' | 'outer' — legacy: pick a voxel *layer* along a
            literal axis 'x'/'y'/'z' (or PCA-yaw-aligned 'u'/'v').
    tolerance_voxels: half-width (in voxels) of the band snapped onto the plane.
    select: optional boolean mask over the grid's voxels to restrict the surface.
    rotation_deg: legacy 'u'/'v' only — override the auto-detected yaw.
    anchor: RANSAC path only — a `PlaneAnchor` to reuse instead of re-fitting,
        so the surface doesn't drift to a different plane/position as the
        voxel size changes (see `PlaneAnchor`).
    """
    if offset_method == "ransac":
        return _smooth_surface_ransac(
            grid, axis, tolerance_voxels, select,
            ransac_threshold=ransac_threshold, ransac_iters=ransac_iters, seed=seed,
            anchor=anchor,
        )

    if anchor is not None:
        raise ValueError("anchor is only supported with offset_method='ransac'")

    if axis in ("u", "v"):
        yaw = principal_yaw(grid, select) if rotation_deg is None else float(rotation_deg)
        rotated, pivot = _rotate_grid_horizontal(grid, yaw)
        inner_axis = "x" if axis == "u" else "y"
        surface = _smooth_surface_core(rotated, inner_axis, offset_method, tolerance_voxels, select)
        _rotate_surface_back(surface, yaw, pivot)
        surface.axis = axis
        surface.rotation_deg = yaw
        return surface

    if rotation_deg is not None:
        raise ValueError("rotation_deg is only valid with axis 'u' or 'v'")
    return _smooth_surface_core(grid, axis, offset_method, tolerance_voxels, select)


def _smooth_surface_ransac(
    grid: VoxelGrid,
    axis: str,
    tolerance_voxels: int = 3,
    select: np.ndarray | None = None,
    ransac_threshold: float | None = None,
    ransac_iters: int = 500,
    seed: int = 0,
    anchor: PlaneAnchor | None = None,
) -> PlanarSurface:
    """Flatten onto a RANSAC-fitted plane, then class-zone it (see smooth_surface).

    The plane is chosen by aligning to a voxel-size-stable target direction
    (`_axis_target`: `u`/`v` from the PCA yaw, `z` vertical), not by picking the
    largest plane of a class — so a fixed axis stays on the same physical wall
    across voxel sizes. `axis='z'` yields an empty surface when the scan has no
    horizontal plane at this resolution.

    `anchor`, if given, skips the fit and axis selection above entirely and
    reuses that exact plane (see `PlaneAnchor`) — the returned surface's
    `normal`/`origin`/`e_u`/`e_v` are populated either way, so the caller can
    capture them (`PlaneAnchor.from_surface`) to anchor the *next* call.
    """
    vs = grid.voxel_size
    mask = np.ones(len(grid), bool) if select is None else np.asarray(select, bool)
    centers = grid.centers[mask]
    labels = grid.labels[mask]
    counts = grid.counts[mask]

    empty = PlanarSurface(axis=axis, plane_coord=0.0, voxel_size=vs)
    if len(centers) < 3:
        return empty

    threshold = ransac_threshold if ransac_threshold else max(0.05, 0.5 * vs)

    if anchor is not None:
        origin, e_u, e_v, n = anchor.origin, anchor.e_u, anchor.e_v, anchor.normal
    else:
        # RANSAC on a random subset (a few thousand voxels are enough to lock
        # onto the dominant planes and keeps interactive smoothing responsive).
        rng = np.random.default_rng(seed)
        if len(centers) > _RANSAC_FIT_CAP:
            fit_pts = centers[rng.choice(len(centers), _RANSAC_FIT_CAP, replace=False)]
        else:
            fit_pts = centers
        min_inliers = max(30, len(fit_pts) // 200)
        planes = extract_planes(
            fit_pts, threshold, iters=ransac_iters, seed=seed,
            max_planes=8, min_inliers=min_inliers,
        )
        # Anchor the axis to a voxel-size-stable target direction (PCA yaw for
        # u/v) and pick the plane best aligned with it, so a fixed axis stays
        # on the same physical wall as the resolution changes.
        yaw = principal_yaw(grid, select)
        target = _axis_target(axis, yaw)
        plane = _select_plane_for_axis(planes, axis, target)
        if plane is None:
            return empty

        # Recompute inliers over *all* selected voxels for a stable in-plane basis.
        full_inl = np.abs((centers - plane.point) @ plane.normal) < threshold
        if full_inl.sum() < 3:
            return empty
        # Canonicalize the normal sign toward the target so the reported normal
        # and the polygon winding don't flip between runs (RANSAC's sign is
        # otherwise arbitrary).
        normal = plane.normal.copy()
        if float(normal @ target) < 0:
            normal = -normal
        origin, e_u, e_v, n = plane_basis(normal, centers[full_inl])

    # Snap every voxel within the tolerance band onto the plane; a recessed
    # window (1-3 voxels deep) lands flush -> a co-planar sub-surface.
    u, v, d = project_to_plane(centers, origin, e_u, e_v, n)
    band = tolerance_voxels * vs
    inlier = np.abs(d) <= band
    deviation = ~inlier

    surface = PlanarSurface(
        axis=axis, plane_coord=float(origin @ n), voxel_size=vs,
        n_inliers=int(inlier.sum()), n_deviations=int(deviation.sum()),
        deviations=VoxelGrid(
            centers=centers[deviation], labels=labels[deviation],
            counts=counts[deviation], voxel_size=vs, origin=grid.origin,
        ),
        rotation_deg=float(np.degrees(np.arctan2(n[1], n[0])) % 180.0),
        normal=n.copy(), origin=origin.copy(), e_u=e_u.copy(), e_v=e_v.copy(),
    )
    if not inlier.any():
        return surface

    # Class-zone the flattened voxels on the in-plane (u, v) lattice (cell =
    # voxel size), then cover each class with greedy rectangles -> quads.
    u0, v0 = float(u[inlier].min()), float(v[inlier].min())
    iu = np.floor((u[inlier] - u0) / vs).astype(np.int64)
    iv = np.floor((v[inlier] - v0) / vs).astype(np.int64)
    raster, i0, j0 = _zone_by_class(np.column_stack([iu, iv]), labels[inlier])
    raster, surface.n_filled, surface.n_unknown = fill_enclosed_cells(raster)

    for cid in np.unique(raster[raster >= 0]):
        polys = []
        for r0, c0, r1, c1 in _greedy_rectangles(raster == cid):
            uu0 = u0 + (i0 + r0) * vs
            uu1 = u0 + (i0 + r1 + 1) * vs
            vv0 = v0 + (j0 + c0) * vs
            vv1 = v0 + (j0 + c1 + 1) * vs
            polys.append(_quad_uv(uu0, uu1, vv0, vv1, origin, e_u, e_v))
        surface.subsurfaces.append(
            SubSurface(int(cid), class_name(int(cid)), _role(int(cid)), polys)
        )
    return surface


# --------------------------------------------------------------------------- #
# Axis-aligned re-projection (opt-in): reuse a RANSAC plane but swap the PCA    #
# basis for one aligned to the world axes, and drop sparse colour blobs.        #
# --------------------------------------------------------------------------- #
def _axis_aligned_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-aligned in-plane axes (e_h, e_w) for a plane of the given normal.

    e_h is the horizontal in-plane direction (world +Z x normal): for a vertical
    facade that is the along-wall horizontal axis, so grid columns come out
    vertical (gravity-aligned). e_w = normal x e_h completes a right-handed frame
    (e_h x e_w = normal), so a CCW winding in (h, w) is CCW about +normal — the
    same convention as `plane_basis`/`_quad_uv`. When the plane is ~horizontal
    (a roof/floor, normal ~ +/-Z) e_h degenerates, so it falls back to
    world +X x normal, giving an X/Y grid.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    e_h = np.cross(np.array([0.0, 0.0, 1.0]), n)
    if np.linalg.norm(e_h) < 1e-6:  # normal ~ +/-Z (roof/floor): use world X instead
        e_h = np.cross(np.array([1.0, 0.0, 0.0]), n)
    e_h = e_h / np.linalg.norm(e_h)
    e_w = np.cross(n, e_h)
    e_w = e_w / np.linalg.norm(e_w)
    return e_h, e_w


def _drop_small_components(
    raster: np.ndarray, cell_size: float, min_side_m: float
) -> np.ndarray:
    """Blank same-class cell blobs whose bounding box is smaller than min_side_m.

    For each class, its cells are grouped into 4-connected components
    (`_label_components`); a component is kept only if the longer side of its
    bounding box (max of width, height) reaches `min_side_m` metres — i.e. a
    single side passing the threshold is enough (logical OR, matching the spec:
    a long thin thermal stripe survives). Components under the threshold (noise,
    isolated specks) are set to -1 (empty). Returns a new raster.
    """
    out = np.asarray(raster).copy()
    if out.size == 0 or min_side_m <= 0:
        return out
    for cid in np.unique(out[out >= 0]):
        labels, n_components = _label_components(out == cid)
        for comp in range(n_components):
            cells = np.argwhere(labels == comp)
            rows, cols = cells[:, 0], cells[:, 1]
            height = (rows.max() - rows.min() + 1) * cell_size
            width = (cols.max() - cols.min() + 1) * cell_size
            if max(height, width) < min_side_m:
                out[rows, cols] = -1
    return out


def project_axis_aligned(
    grid: VoxelGrid,
    surface: PlanarSurface,
    min_side_m: float = 1.0,
    tolerance_voxels: int = 3,
    select: np.ndarray | None = None,
    cell_size: float | None = None,
) -> PlanarSurface:
    """Re-raster a RANSAC surface on a world-axis-aligned grid, with a size gate.

    `surface` must come from the RANSAC path (its `normal`/`origin` are set). The
    fitted plane is reused as-is (no re-fit): the same voxels are re-projected
    onto it, but the in-plane grid is built on the world-aligned basis
    (`_axis_aligned_basis`) instead of the PCA basis, so it follows the world
    axes (vertical columns on a facade, X/Y on a roof) rather than a diagonal.
    Each class is then zoned as usual, but same-class cell blobs whose bounding
    box is under `min_side_m` on *both* sides are dropped as noise
    (`_drop_small_components`) — so only colours that are "sufficiently present"
    (>= min_side_m along at least one side) are carried onto this second plane.

    `cell_size` defaults to the surface's voxel size. The tolerance band reuses
    the surface's voxel size (as the first pass did). Returns a new
    `PlanarSurface`; the input is left untouched.
    """
    if surface.normal is None or surface.origin is None:
        raise ValueError(
            "project_axis_aligned needs a RANSAC-fitted surface "
            "(normal/origin set) — run smooth_surface with offset_method='ransac'"
        )
    n = np.asarray(surface.normal, dtype=np.float64)
    origin = np.asarray(surface.origin, dtype=np.float64)
    cell = float(cell_size) if cell_size else surface.voxel_size
    e_h, e_w = _axis_aligned_basis(n)

    mask = np.ones(len(grid), bool) if select is None else np.asarray(select, bool)
    centers = grid.centers[mask]
    labels = grid.labels[mask]

    out = PlanarSurface(
        axis=surface.axis, plane_coord=float(origin @ n), voxel_size=cell,
        rotation_deg=surface.rotation_deg,
        normal=n.copy(), origin=origin.copy(), e_u=e_h.copy(), e_v=e_w.copy(),
    )
    if len(centers) < 1:
        return out

    u, v, d = project_to_plane(centers, origin, e_h, e_w, n)
    band = tolerance_voxels * surface.voxel_size
    inlier = np.abs(d) <= band
    out.n_inliers = int(inlier.sum())
    out.n_deviations = int((~inlier).sum())
    if not inlier.any():
        return out

    u0, v0 = float(u[inlier].min()), float(v[inlier].min())
    iu = np.floor((u[inlier] - u0) / cell).astype(np.int64)
    iv = np.floor((v[inlier] - v0) / cell).astype(np.int64)
    raster, i0, j0 = _zone_by_class(np.column_stack([iu, iv]), labels[inlier])
    raster = _drop_small_components(raster, cell, min_side_m)

    for cid in np.unique(raster[raster >= 0]):
        polys = []
        for r0, c0, r1, c1 in _greedy_rectangles(raster == cid):
            uu0 = u0 + (i0 + r0) * cell
            uu1 = u0 + (i0 + r1 + 1) * cell
            vv0 = v0 + (j0 + c0) * cell
            vv1 = v0 + (j0 + c1 + 1) * cell
            polys.append(_quad_uv(uu0, uu1, vv0, vv1, origin, e_h, e_w))
        out.subsurfaces.append(
            SubSurface(int(cid), class_name(int(cid)), _role(int(cid)), polys)
        )
    return out


def _smooth_surface_core(
    grid: VoxelGrid,
    axis: str,
    offset_method: str = "mode",
    tolerance_voxels: int = 3,
    select: np.ndarray | None = None,
) -> PlanarSurface:
    """Legacy literal-axis flatten (mode/median/outer layer); see smooth_surface."""
    if axis not in _AXES:
        raise ValueError(f"axis must be one of x, y, z (got {axis!r})")
    a = _AXES[axis]
    u_idx, v_idx = [i for i in range(3) if i != a]  # the two in-plane axes
    vs = grid.voxel_size

    layers = _layer_indices(grid)
    mask = np.ones(len(grid), bool) if select is None else np.asarray(select, bool)
    sel_layers = layers[mask]

    plane_layer = _plane_layer(sel_layers[:, a], offset_method)
    plane_coord = grid.origin[a] + (plane_layer + 0.5) * vs

    # Inliers: within the tolerance band around the plane (and in `select`).
    dist = np.abs(layers[:, a] - plane_layer)
    inlier = mask & (dist <= tolerance_voxels)
    deviation = mask & ~inlier

    surface = PlanarSurface(
        axis=axis, plane_coord=float(plane_coord), voxel_size=vs,
        n_inliers=int(inlier.sum()), n_deviations=int(deviation.sum()),
        deviations=_subset(grid, deviation),
    )
    if not inlier.any():
        return surface

    # Project inliers to the 2-D in-plane lattice and take a majority class per
    # cell, then cover each class with greedy rectangles.
    uv = layers[inlier][:, [u_idx, v_idx]]
    raster, u0g, v0g = _zone_by_class(uv, grid.labels[inlier])
    raster, surface.n_filled, surface.n_unknown = fill_enclosed_cells(raster)

    origin_u = grid.origin[u_idx]
    origin_v = grid.origin[v_idx]
    for cid in np.unique(raster[raster >= 0]):
        rects = _greedy_rectangles(raster == cid)
        polys = []
        for r0, c0, r1, c1 in rects:
            # cell (r) spans world u in [origin_u + (u0g+r)*vs, +1 cell]
            uu0 = origin_u + (u0g + r0) * vs
            uu1 = origin_u + (u0g + r1 + 1) * vs
            vv0 = origin_v + (v0g + c0) * vs
            vv1 = origin_v + (v0g + c1 + 1) * vs
            polys.append(_quad(uu0, uu1, vv0, vv1, plane_coord, a, u_idx, v_idx))
        surface.subsurfaces.append(
            SubSurface(int(cid), class_name(int(cid)), _role(int(cid)), polys)
        )
    return surface


def _subset(grid: VoxelGrid, mask: np.ndarray) -> VoxelGrid:
    return VoxelGrid(
        centers=grid.centers[mask], labels=grid.labels[mask],
        counts=grid.counts[mask], voxel_size=grid.voxel_size, origin=grid.origin,
    )


def to_openstudio_json(surface: PlanarSurface, path: str | Path) -> Path:
    """Write the planar surface as OpenStudio-friendly JSON (polygons + roles).

    Schema: {axis, plane_coord, rotation_deg, voxel_size, normal, surfaces:
    [{class_id, class_name, role, polygons: [[[x,y,z], x4], ...]}],
    n_inliers, n_deviations}. Polygon vertices are in world coordinates and are
    planar by construction (RANSAC: they share the fitted plane; legacy x/y/z:
    they share `plane_coord`; legacy u/v: they share it once rotated by
    -rotation_deg back into the building-aligned frame). Wound CCW about the
    +normal. `role` maps envelope classes to a base Surface and fenestration
    classes to a SubSurface downstream. `normal` is the unit plane normal for
    the RANSAC fit, else null.
    """
    doc = {
        "axis": surface.axis,
        "plane_coord": surface.plane_coord,
        "rotation_deg": surface.rotation_deg,
        "voxel_size": surface.voxel_size,
        "normal": None if surface.normal is None else np.asarray(surface.normal).tolist(),
        "n_inliers": surface.n_inliers,
        "n_deviations": surface.n_deviations,
        "surfaces": [
            {
                "class_id": s.class_id,
                "class_name": s.class_name,
                "role": s.role,
                "polygons": [poly.tolist() for poly in s.polygons],
            }
            for s in surface.subsurfaces
        ],
    }
    path = Path(path)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
