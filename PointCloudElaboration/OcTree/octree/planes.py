"""Dominant-plane detection and per-wall 2D rasterization (RANSAC/MSAC).

This is the point-cloud side of turning a stepped voxel *shell* into a flat,
continuous per-wall raster, computed in-plane rather than by reconstructing a
3-D mesh from occupancy. The pipeline (see the module README):

1. Ground removal — the *known-vertical-normal* trick (like the calibration
   board): the ground is a horizontal slab, so fix the normal to +z and take
   the densest low-z layer as its offset (`detect_ground`). Optionally also
   drop ground/terrain semantic classes (`ground_mask_from_labels`).
2. Dominant plane — RANSAC/MSAC on the raw points *or* the voxel centroids
   (a faster proxy): sample 3 points, score inliers, keep the best, refit the
   normal to all inliers via SVD (`fit_plane_ransac`). One plane, the one with
   the most inliers.
3. Local 2-D basis — from the plane normal, PCA on the inliers gives two
   orthonormal in-plane axes (u, v) aligned with the wall (`plane_basis`), a
   3-D generalization of smoothing.principal_yaw.
4. Projection — every voxel/point gets an in-plane (u, v) and a signed
   perpendicular offset d (`project_to_plane`). d is QC only: it flags
   protrusions / mis-detections and is *not* folded back into the geometry.
5. Raster — bin (u, v) into a regular grid and average the per-point scalar
   (temperature) per cell (`rasterize`). This is the "smoothing" result.
6. Wall polygon — the minimum-area rotated rectangle of the (u, v) footprint
   (`min_area_rect`), mapped back to 3-D; it becomes the OpenStudio Surface
   (`wall_to_surface`, reusing smoothing.PlanarSurface).

Pure numpy on purpose (no sklearn/open3d/scipy, and no pyvista): RANSAC, the
convex hull and the rotating-calipers rectangle are all implemented here, so
the pipeline and its self-test run headless. Rendering lives in viewer.py.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classes import class_name
from .smoothing import PlanarSurface, SubSurface

# Semantic ids treated as ground when labels are available (classes.py).
_GROUND_CLASS_IDS = (11, 12)  # ground surface, terrain

# Wall class id used for the exported envelope surface (classes.py: 1 == wall).
_WALL_CLASS_ID = 1


# --------------------------------------------------------------------------- #
# Per-point scalar (temperature) — synthetic fallback for the current sample  #
# --------------------------------------------------------------------------- #
def synthetic_temperature(points: np.ndarray, seed: int = 0) -> np.ndarray:
    """A deterministic, smooth per-point temperature field (deg C) for demos.

    The point cloud carries no temperature yet (LiDAR<->thermal co-registration
    is future work), so this stands in: a gentle gradient with height plus a
    horizontal ripple and mild noise, enough for the raster to show structure.
    Deterministic in `seed` so screenshots and the self-test are reproducible.
    """
    p = np.asarray(points, dtype=np.float64)
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    n = (p - lo) / span  # normalized 0..1 per axis
    rng = np.random.default_rng(seed)
    field = (
        12.0
        + 10.0 * n[:, 2]
        + 4.0 * np.sin(2.0 * np.pi * 2.0 * n[:, 0])
        + 3.0 * np.cos(2.0 * np.pi * 2.0 * n[:, 1])
    )
    field += rng.normal(0.0, 0.4, size=len(p))
    return field


# --------------------------------------------------------------------------- #
# Ground removal                                                              #
# --------------------------------------------------------------------------- #
def detect_ground(
    points: np.ndarray,
    band: float = 0.5,
    z_search_frac: float = 0.3,
    min_frac: float = 0.02,
) -> np.ndarray:
    """Boolean mask of a near-horizontal ground slab, via a fixed vertical normal.

    The ground's normal is known (vertical), so only z matters: histogram the
    lowest `z_search_frac` of the height range at ~`band` resolution and take
    the densest layer as the ground offset z0; points within +/-`band` of z0
    are ground. Returns all-False if the densest low layer holds less than
    `min_frac` of the cloud (no clear ground present).
    """
    z = np.asarray(points, dtype=np.float64)[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    zspan = zmax - zmin
    if zspan <= 0:
        return np.zeros(len(z), dtype=bool)

    hi = zmin + z_search_frac * zspan
    lo_region = z[z <= hi]
    nbins = max(1, int(np.ceil((hi - zmin) / max(band, 1e-6))))
    counts, edges = np.histogram(lo_region, bins=nbins, range=(zmin, hi))
    peak = int(counts.argmax())
    z0 = 0.5 * (edges[peak] + edges[peak + 1])

    ground = np.abs(z - z0) <= band
    if ground.mean() < min_frac:
        return np.zeros(len(z), dtype=bool)
    return ground


def ground_mask_from_labels(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of points whose semantic class is ground/terrain."""
    labels = np.asarray(labels)
    mask = np.zeros(len(labels), dtype=bool)
    for cid in _GROUND_CLASS_IDS:
        mask |= labels == cid
    return mask


# --------------------------------------------------------------------------- #
# RANSAC / MSAC plane fit                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Plane:
    normal: np.ndarray  # (3,) unit normal
    point: np.ndarray  # (3,) a point on the plane (inlier centroid after refit)
    inliers: np.ndarray  # (N,) bool over the fitted points


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float,
    iters: int = 1000,
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
    iters: int = 1000,
    seed: int = 0,
    cost: str = "msac",
    max_planes: int = 8,
    min_inliers: int = 200,
) -> list[Plane]:
    """Iteratively RANSAC-fit planes: fit, strip inliers, repeat.

    Returns up to `max_planes` planes, each with an `inliers` mask over the
    *original* `points`, sorted by inlier count (descending). Stops early when
    the next plane would have fewer than `min_inliers`. This is how the caller
    offers a *choice* of wall (rank / target-normal) instead of only the single
    most-populated plane, and the first step toward full multi-wall extraction.
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


def _select_plane(
    planes: list[Plane],
    rank: int,
    target_normal=None,
    orientation: str = "any",
) -> tuple[int, Plane]:
    """Pick a plane from `planes` by orientation filter + target-normal or rank.

    orientation: 'any' | 'vertical' (facades, |n_z|<0.3) | 'horizontal'
    (floors/roofs, |n_z|>0.7). target_normal (if given) picks the best-aligned
    plane; otherwise the 1-based `rank` by inlier count. Returns (index_in_all,
    plane) so callers can report "plane R of N".
    """
    if not planes:
        raise RuntimeError("no planes found")
    order = list(range(len(planes)))
    if orientation == "vertical":
        order = [i for i in order if abs(planes[i].normal[2]) < 0.3]
    elif orientation == "horizontal":
        order = [i for i in order if abs(planes[i].normal[2]) > 0.7]
    if not order:
        raise RuntimeError(
            f"no {orientation} plane among the {len(planes)} detected "
            "(try --orientation any, or a larger --ransac-threshold)"
        )

    if target_normal is not None:
        t = np.asarray(target_normal, dtype=np.float64)
        t = t / np.linalg.norm(t)
        dots = {i: abs(planes[i].normal @ t) for i in order}
        # Among planes within ~37 deg of the target direction, take the biggest
        # (planes is sorted by inliers), so we get the *main* wall facing that
        # way, not a tiny sliver that happens to be marginally better aligned.
        # A wide cone is safe: the two facade families are ~90 deg apart, and it
        # tolerates the building yaw / an approximate target from the user.
        within = [i for i in order if dots[i] >= 0.80]
        best = within[0] if within else max(order, key=lambda i: dots[i])
        return best, planes[best]

    r = int(np.clip(rank - 1, 0, len(order) - 1))
    return order[r], planes[order[r]]


# --------------------------------------------------------------------------- #
# Local 2-D basis on the plane                                                #
# --------------------------------------------------------------------------- #
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
    centroid (a point on the plane). Mirrors smoothing.principal_yaw, in 3-D.
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


# --------------------------------------------------------------------------- #
# Rasterization                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Raster:
    values: np.ndarray  # (nu, nv) mean scalar per cell, NaN where empty
    counts: np.ndarray  # (nu, nv) points/voxels per cell
    offset: np.ndarray  # (nu, nv) mean |d| per cell (QC), NaN where empty
    cell_size: float
    u0: float  # world u of cell (0, *) lower edge
    v0: float  # world v of cell (*, 0) lower edge

    @property
    def shape(self):
        return self.values.shape

    @property
    def occupancy(self) -> float:
        return float((self.counts > 0).mean()) if self.counts.size else 0.0


def rasterize(u, v, values, cell_size, offsets=None) -> Raster:
    """Bin (u, v) into a regular grid; per cell, mean of `values` and mean |d|.

    Empty cells are NaN. `values` may be None (no scalar) -> the value grid is
    all-NaN but the count/offset grids are still filled.
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u0, v0 = float(u.min()), float(v.min())
    iu = np.floor((u - u0) / cell_size).astype(np.int64)
    iv = np.floor((v - v0) / cell_size).astype(np.int64)
    nu = int(iu.max()) + 1
    nv = int(iv.max()) + 1

    counts = np.zeros((nu, nv), dtype=np.int64)
    np.add.at(counts, (iu, iv), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        if values is not None:
            vsum = np.zeros((nu, nv), dtype=np.float64)
            np.add.at(vsum, (iu, iv), np.asarray(values, dtype=np.float64))
            vmean = np.where(counts > 0, vsum / counts, np.nan)
        else:
            vmean = np.full((nu, nv), np.nan)

        if offsets is not None:
            osum = np.zeros((nu, nv), dtype=np.float64)
            np.add.at(osum, (iu, iv), np.abs(np.asarray(offsets, dtype=np.float64)))
            omean = np.where(counts > 0, osum / counts, np.nan)
        else:
            omean = np.full((nu, nv), np.nan)

    return Raster(values=vmean, counts=counts, offset=omean,
                  cell_size=float(cell_size), u0=u0, v0=v0)


# --------------------------------------------------------------------------- #
# Minimum-area rotated rectangle (convex hull + rotating calipers)           #
# --------------------------------------------------------------------------- #
def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """CCW convex hull of 2-D points (Andrew's monotone chain), pure numpy."""
    pts = np.unique(np.asarray(pts, dtype=np.float64), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _bbox_corners(uv: np.ndarray) -> np.ndarray:
    u0, v0 = uv.min(axis=0)
    u1, v1 = uv.max(axis=0)
    return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])


def min_area_rect(uv: np.ndarray) -> np.ndarray:
    """Four corners (4, 2), CCW, of the minimum-area rotated rectangle of uv.

    Rotating calipers over the convex-hull edges: the min-area rectangle has a
    side collinear with a hull edge, so test each edge's frame and keep the
    smallest bounding box. Falls back to the axis-aligned bbox if degenerate.
    """
    uv = np.asarray(uv, dtype=np.float64)
    hull = _convex_hull(uv)
    if len(hull) < 3:
        return _bbox_corners(uv)

    best = None
    m = len(hull)
    for i in range(m):
        edge = hull[(i + 1) % m] - hull[i]
        length = np.linalg.norm(edge)
        if length < 1e-12:
            continue
        ex = edge / length
        ey = np.array([-ex[1], ex[0]])
        px = hull @ ex
        py = hull @ ey
        x0, x1 = px.min(), px.max()
        y0, y1 = py.min(), py.max()
        area = (x1 - x0) * (y1 - y0)
        if best is None or area < best[0]:
            best = (area, ex, ey, x0, x1, y0, y1)

    _, ex, ey, x0, x1, y0, y1 = best
    return np.array([
        x0 * ex + y0 * ey,
        x1 * ex + y0 * ey,
        x1 * ex + y1 * ey,
        x0 * ex + y1 * ey,
    ])


# --------------------------------------------------------------------------- #
# Orchestration                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class WallPlane:
    normal: np.ndarray
    origin: np.ndarray  # a point on the plane (inlier centroid)
    e_u: np.ndarray
    e_v: np.ndarray
    n_inliers: int
    n_fitted: int  # points RANSAC ran on (after ground removal)
    n_ground: int  # points dropped as ground
    rect_uv: np.ndarray  # (4, 2) rectangle corners in the (u, v) frame
    rect_xyz: np.ndarray  # (4, 3) same corners in world coordinates
    raster: Raster
    offset_stats: dict = field(default_factory=dict)
    cell_size: float = 0.0
    rank: int = 1  # 1-based index of this plane among the detected candidates
    n_candidates: int = 1  # how many planes were detected (for cycling / choice)

    @property
    def rect_dims(self) -> tuple[float, float]:
        """(width, height) of the wall rectangle in metres."""
        w = float(np.linalg.norm(self.rect_uv[1] - self.rect_uv[0]))
        h = float(np.linalg.norm(self.rect_uv[3] - self.rect_uv[0]))
        return w, h


def run_dominant_plane(
    points: np.ndarray,
    values: np.ndarray | None = None,
    *,
    threshold: float = 0.10,
    iters: int = 1000,
    seed: int = 0,
    cost: str = "msac",
    keep_ground: bool = False,
    ground_band: float = 0.5,
    raster_cell: float = 0.10,
    labels: np.ndarray | None = None,
    protrusion_factor: float = 4.0,
    qc_depth: float | None = None,
    rank: int = 1,
    target_normal=None,
    orientation: str = "any",
    max_planes: int = 8,
) -> WallPlane:
    """Full pipeline: ground removal -> RANSAC -> basis -> raster -> rectangle.

    `values` is the per-point scalar averaged into the raster (temperature).
    `labels`, if given, lets ground removal also drop ground/terrain classes.

    Plane choice: buildings have several large planes (e.g. two perpendicular
    facade families). Instead of always taking the single most-populated plane,
    `extract_planes` finds the top candidates and one is picked by `orientation`
    ('any'/'vertical'/'horizontal') plus either `target_normal` (best-aligned)
    or the 1-based `rank` by inlier count (rank=2 -> the next-biggest wall, i.e.
    the *other* facade). QC: the perpendicular offset d is summarized over the
    voxels within this wall's footprint and a shallow depth window `qc_depth`
    (default max(1 m, 10*threshold)); points beyond `protrusion_factor *
    threshold` are counted as protrusions.
    """
    pts = np.asarray(points, dtype=np.float64)
    vals = None if values is None else np.asarray(values, dtype=np.float64)

    keep = np.ones(len(pts), dtype=bool)
    if not keep_ground:
        ground = detect_ground(pts, band=ground_band)
        if labels is not None:
            ground = ground | ground_mask_from_labels(labels)
        keep = ~ground
    n_ground = int((~keep).sum())

    work = pts[keep]
    work_vals = None if vals is None else vals[keep]
    if len(work) < 3:
        raise RuntimeError("too few points left after ground removal")

    planes = extract_planes(work, threshold, iters, seed, cost, max_planes=max_planes)
    sel_i, plane = _select_plane(planes, rank, target_normal, orientation)
    inl_pts = work[plane.inliers]
    origin, e_u, e_v, n = plane_basis(plane.normal, inl_pts)

    # Raster over the wall inliers.
    u, v, d = project_to_plane(inl_pts, origin, e_u, e_v, n)
    inl_vals = None if work_vals is None else work_vals[plane.inliers]
    raster = rasterize(u, v, inl_vals, raster_cell, offsets=d)

    # Wall rectangle from the (u, v) footprint, mapped back to 3-D.
    rect_uv = min_area_rect(np.column_stack([u, v]))
    rect_xyz = (
        origin[None, :]
        + rect_uv[:, 0:1] * e_u[None, :]
        + rect_uv[:, 1:2] * e_v[None, :]
    )

    # QC scoped to this wall: kept voxels whose (u, v) falls inside the inlier
    # footprint and within a shallow depth window (so other facades/roof, which
    # lie far off this plane, are not miscounted as protrusions).
    u_all, v_all, d_all = project_to_plane(work, origin, e_u, e_v, n)
    umin, umax, vmin, vmax = u.min(), u.max(), v.min(), v.max()
    depth = qc_depth if qc_depth is not None else max(1.0, 10.0 * threshold)
    in_fp = (u_all >= umin) & (u_all <= umax) & (v_all >= vmin) & (v_all <= vmax)
    scope = in_fp & (np.abs(d_all) <= depth)
    d_abs = np.abs(d_all[scope])
    pb = protrusion_factor * threshold
    if len(d_abs) == 0:
        d_abs = np.zeros(1)
    offset_stats = {
        "n_footprint": int(scope.sum()),
        "qc_depth_m": float(depth),
        "d_mean_abs_m": float(d_abs.mean()),
        "d_p95_abs_m": float(np.percentile(d_abs, 95)),
        "d_max_abs_m": float(d_abs.max()),
        "protrusion_band_m": float(pb),
        "n_protrusions": int((d_abs > pb).sum()),
        "protrusion_frac": float((d_abs > pb).mean()),
    }

    return WallPlane(
        normal=n, origin=origin, e_u=e_u, e_v=e_v,
        n_inliers=int(plane.inliers.sum()), n_fitted=int(len(work)),
        n_ground=n_ground, rect_uv=rect_uv, rect_xyz=rect_xyz,
        raster=raster, offset_stats=offset_stats, cell_size=float(raster_cell),
        rank=sel_i + 1, n_candidates=len(planes),
    )


# --------------------------------------------------------------------------- #
# Export                                                                      #
# --------------------------------------------------------------------------- #
def wall_to_surface(wall: WallPlane) -> PlanarSurface:
    """Wrap the wall rectangle as a PlanarSurface (one envelope sub-surface).

    Reuses smoothing.PlanarSurface/SubSurface so the rectangle flows through
    the existing to_openstudio_json / openstudio_adapter.to_osm unchanged: a
    single wall-class quad becomes one base Surface. `axis='n'` marks a
    general (RANSAC) plane; plane_coord is the plane's signed offset along its
    normal.
    """
    quad = np.asarray(wall.rect_xyz, dtype=np.float64)
    sub = SubSurface(
        class_id=_WALL_CLASS_ID, class_name=class_name(_WALL_CLASS_ID),
        role="envelope", polygons=[quad],
    )
    return PlanarSurface(
        axis="n",
        plane_coord=float(wall.origin @ wall.normal),
        voxel_size=float(wall.cell_size),
        subsurfaces=[sub],
        n_inliers=int(wall.n_inliers),
        n_deviations=int(wall.offset_stats.get("n_protrusions", 0)),
    )


def wall_qc_dict(wall: WallPlane) -> dict:
    """Serializable plane parameters + QC (offset-d stats) for a sidecar JSON."""
    w, h = wall.rect_dims
    return {
        "normal": wall.normal.tolist(),
        "origin": wall.origin.tolist(),
        "e_u": wall.e_u.tolist(),
        "e_v": wall.e_v.tolist(),
        "n_inliers": int(wall.n_inliers),
        "n_fitted": int(wall.n_fitted),
        "n_ground": int(wall.n_ground),
        "inlier_frac": float(wall.n_inliers / wall.n_fitted) if wall.n_fitted else 0.0,
        "rect_width_m": w,
        "rect_height_m": h,
        "rect_area_m2": w * h,
        "raster_shape": list(wall.raster.shape),
        "raster_cell_m": float(wall.cell_size),
        "raster_occupancy": wall.raster.occupancy,
        "offset_qc": wall.offset_stats,
    }


def save_raster_npy(wall: WallPlane, path: str | Path) -> Path:
    """Save the raster's mean-value grid (NaN = empty) as a .npy array."""
    path = Path(path)
    np.save(path, wall.raster.values)
    return path
