"""Surface smoothing: flatten stepped voxels into planar OpenStudio surfaces.

OpenStudio / EnergyPlus need planar, well-formed surfaces; the raw voxel wall
is stepped. `smooth_surface` projects the voxels of one surface onto a single
plane along a chosen axis (the surface normal), keeping each voxel's semantic
class so the flattened wall is subdivided into homogeneous sub-surfaces
(wall, window, door, ...).

Design (see the module README):
- Plane offset along the axis = the *mode* voxel layer (the most-populated
  layer) by default — robust to noise and to a minority of recessed/protruding
  voxels, and grid-aligned. `median` and `outer` (95th-pct exterior face) are
  options.
- A tolerance band (+/- tolerance_voxels) decides membership: voxels within the
  band snap onto the plane (a recessed window is only 1-3 voxels deep, so it
  snaps flush -> a co-planar sub-surface, which is what OpenStudio requires for
  fenestration); voxels beyond the band are returned as `deviations` (kept, not
  dropped) so the caller can treat them as noise or as their own surface.

Axes:
- `'x' | 'y' | 'z'` — literal world axes (unrotated).
- `'u' | 'v'` — auto-aligned horizontal axes: `principal_yaw` finds the
  building's actual yaw via PCA on the horizontal footprint (real buildings
  are rarely aligned with world x/y), `u` flattens along the dominant wall
  direction and `v` along the perpendicular one (the "other" facade). This
  only corrects yaw (rotation about vertical z); a genuinely pitched/rolled
  scan, or a building with more than two wall directions, needs a fuller
  multi-plane segmentation (documented as future work).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classes import MAX_CLASS_ID, class_name
from .voxelizer import VoxelGrid

_AXES = {"x": 0, "y": 1, "z": 2}

# Semantic classes treated as building envelope (base Surface) vs fenestration
# (SubSurface) when exporting to OpenStudio. Ids follow classes.py.
_FENESTRATION_IDS = {2, 3, 14}  # window, door, blinds
_ENVELOPE_ROLE = "envelope"
_FENESTRATION_ROLE = "fenestration"


@dataclass
class SubSurface:
    class_id: int
    class_name: str
    role: str  # "envelope" | "fenestration"
    polygons: list[np.ndarray]  # each (4, 3) planar quad, world coords, CCW about +normal


@dataclass
class PlanarSurface:
    axis: str  # 'x' | 'y' | 'z' | 'u' | 'v' — the surface-normal axis (flattened)
    plane_coord: float  # coordinate of the plane along `axis` (world for x/y/z;
    #                      along the rotated u/v direction otherwise)
    voxel_size: float
    subsurfaces: list[SubSurface] = field(default_factory=list)
    n_inliers: int = 0
    n_deviations: int = 0
    deviations: VoxelGrid | None = None  # voxels beyond the tolerance band, world coords
    rotation_deg: float = 0.0  # yaw applied for 'u'/'v'; 0.0 for literal x/y/z

    @property
    def n_polygons(self) -> int:
        return sum(len(s.polygons) for s in self.subsurfaces)


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
    raise ValueError(f"Unknown offset_method '{method}' (mode | median | outer)")


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


def _quad(u0, u1, v0, v1, plane_coord, axis_idx, u_idx, v_idx):
    """Planar quad (4,3) for in-plane cell-corner spans [u0,u1] x [v0,v1]."""
    corners_uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]  # CCW in the u-v plane
    quad = np.empty((4, 3), dtype=float)
    for row, (u, v) in enumerate(corners_uv):
        quad[row, axis_idx] = plane_coord
        quad[row, u_idx] = u
        quad[row, v_idx] = v
    return quad


def smooth_surface(
    grid: VoxelGrid,
    axis: str,
    offset_method: str = "mode",
    tolerance_voxels: int = 3,
    select: np.ndarray | None = None,
    rotation_deg: float | None = None,
) -> PlanarSurface:
    """Flatten `grid` onto a single plane along `axis`, preserving class zoning.

    axis: 'x' | 'y' | 'z' (literal world axes) or 'u' | 'v' (auto-aligned:
        dominant / perpendicular horizontal wall direction, via principal_yaw).
    offset_method: 'mode' (default) | 'median' | 'outer'.
    tolerance_voxels: half-width (in voxels) of the band snapped onto the plane.
    select: optional boolean mask over the grid's voxels to restrict the surface.
    rotation_deg: only valid with axis 'u'/'v' — override the auto-detected yaw
        (e.g. for a near-square footprint where PCA might pick either direction).
    """
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


def _smooth_surface_core(
    grid: VoxelGrid,
    axis: str,
    offset_method: str = "mode",
    tolerance_voxels: int = 3,
    select: np.ndarray | None = None,
) -> PlanarSurface:
    """Literal-axis flatten (the pre-rotation logic); see smooth_surface."""
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
    # cell (reusing the histogram idea from voxelizer._grid_from_index).
    uv = layers[inlier][:, [u_idx, v_idx]]
    labels_in = grid.labels[inlier]
    keys, inverse = np.unique(uv, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    hist = np.zeros((len(keys), MAX_CLASS_ID + 1), np.int64)
    np.add.at(hist, (inverse, np.clip(labels_in, 0, MAX_CLASS_ID)), 1)
    cell_labels = hist.argmax(axis=1)

    # Rasterize into a dense 2-D class grid (-1 = empty) for rectangle covering.
    u0g, v0g = keys[:, 0].min(), keys[:, 1].min()
    n_u = keys[:, 0].max() - u0g + 1
    n_v = keys[:, 1].max() - v0g + 1
    raster = np.full((n_u, n_v), -1, np.int64)
    raster[keys[:, 0] - u0g, keys[:, 1] - v0g] = cell_labels

    origin_u = grid.origin[u_idx]
    origin_v = grid.origin[v_idx]
    for cid in np.unique(cell_labels):
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
        values=None if grid.values is None else grid.values[mask],
    )


def to_openstudio_json(surface: PlanarSurface, path: str | Path) -> Path:
    """Write the planar surface as OpenStudio-friendly JSON (polygons + roles).

    Schema: {axis, plane_coord, rotation_deg, voxel_size, surfaces:
    [{class_id, class_name, role, polygons: [[[x,y,z], x4], ...]}],
    n_inliers, n_deviations}. Polygon vertices are in world coordinates and
    are planar by construction: for axis x/y/z they share `plane_coord`
    directly; for axis u/v they share `plane_coord` once rotated by
    `-rotation_deg` back into the building-aligned frame (they were rotated
    into world coordinates for storage/rendering). Wound CCW about the
    +axis normal (in the building-aligned frame for u/v). `role` maps
    envelope classes to a base Surface and fenestration classes to a
    SubSurface downstream.
    """
    doc = {
        "axis": surface.axis,
        "plane_coord": surface.plane_coord,
        "rotation_deg": surface.rotation_deg,
        "voxel_size": surface.voxel_size,
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
