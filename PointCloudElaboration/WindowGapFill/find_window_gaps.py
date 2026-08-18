"""Find window openings as HOLES in wall voxel coverage, and fill them with
synthetic "glass" voxels.

Supersedes the material-vote approach (MaterialToVoxel's material_id ==
"glass"): real window glass barely returns LiDAR at all (specular/
transparent), so a window is much more reliably a literal ABSENCE of
occupied voxels on a wall than a sparse cluster of "glass"-classified ones
-- on the one real check that motivated this folder, the material-vote
approach found only 2 small clusters (13, 22 voxels) vs. this approach's 4
holes (63-82 cells each, ~1-2 m^2, at a consistent height band across BOTH
independently-processed walls -- a strong physical sanity check the
material-vote result didn't have).

Method, per wall (north/south only -- see find_target_walls):
  1. Take every voxel within --wall-tol-m of the wall's own RANSAC plane
     (AlignedOctree/planes_aligned.json) -- its "near-wall structure",
     including reveal/frame depth, not just exact plane inliers.
  2. Rasterize those voxels' (x, z) into a 2D occupancy grid (the wall is
     Y-normal on this corridor, so x/z ARE already sensible in-plane axes --
     no PCA basis needed, unlike OcTree/smoothing.py's generic-orientation
     case).
  3. Light morphological closing (scipy.ndimage.binary_closing, 3x3,
     --closing-iterations) BEFORE the hole test. Real wall coverage at
     15cm is naturally porous (radiators, pillars, uneven returns -- see
     EmissivityCalculation/classify_session.py's docstring on this
     corridor's own wall clutter) -- tried without closing first: 0 holes
     found, because that porosity lets the background bleed to the raster's
     own border almost everywhere (same failure TemperatureToVoxel/
     FILLER.md hit for floor/ceiling gaps, step 1). Closing bridges that
     small-scale porosity while a real window (meters across) stays open.
     Empirically checked (this corridor, 15cm grid, --wall-tol-m 0.30):
     iterations=1 leaves a spurious ~9-cell speck; iterations>=3 over-closes
     and erases the real holes too, collapsing to 1 connected component with
     nothing left inside it. 2 is the sweet spot -- NOT validated for other
     voxel sizes/geometries, retune if the grid changes materially.
  4. Label the CLOSED grid's background (scipy.ndimage.label); any
     component that does NOT touch the raster's own bounding-box border is
     a true enclosed hole (unlike FILLER.md's floor/ceiling case, a wall's
     true physical extent -- bounded by floor/ceiling/end-walls, all
     opaque/reflective, well-captured surfaces -- makes "enclosed within
     THIS wall's own footprint" the physically correct test here, not the
     coverage-edge artifact it was there).
  5. Size-gate (--min-hole-cells / --max-hole-cells): drop dropout specks
     and anything implausibly large (probably not a window).
  6. Drop holes reaching within --floor-margin-m of the floor (a floor-to-
     somewhere opening is a doorway/passage, not an elevated window) --
     FILLER.md's "what to try next" #2 (explicit distance-to-wall-plane /
     distance-to-floor check), applied here instead of the count-ratio
     heuristic that didn't generalize there.
  7. Every surviving hole cell becomes one synthetic voxel, placed AT the
     wall's own plane (x, wall_y, z) -- not at whatever depth the raw
     points happened to sit, since there mostly aren't any.

Usage:
    python find_window_gaps.py
        [--voxels ../AlignedOctree/voxels.npz]
        [--planes-aligned ../AlignedOctree/planes_aligned.json]
        [--wall-tol-m 0.30] [--floor-margin-m 0.30]
        [--closing-iterations 2] [--min-hole-cells 6] [--max-hole-cells 400]
        [--out window_gaps.npz] [--windows-out windows.json]

Thresholds above are tuned for a 15cm voxels.npz (AlignedOctree's default).
A materially different voxel size needs retuning -- nothing here rescales
them automatically.

Venv: C:\\venvs\\planefit (numpy, scipy only).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

WALL_TOL_M = 0.30
FLOOR_MARGIN_M = 0.30
CLOSING_ITERATIONS = 2
MIN_HOLE_CELLS = 6
MAX_HOLE_CELLS = 400

BACKGROUND_STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)  # 4-connected
CLOSING_STRUCTURE = np.ones((3, 3), dtype=bool)


def axis_of(plane):
    """Index (0/1/2 = X/Y/Z) of the world axis a plane's normal is nearest
    to -- same convention as AlignedOctree/aligned_octree.py's _axis_of
    (copied, not imported: this folder is self-contained)."""
    return int(np.argmax(np.abs(np.asarray(plane["normal"]))))


def find_target_walls(planes):
    """The wall pair whose normal is closest to world Y -- "north/south" on
    this corridor's aligned frame (AlignedOctree's building-frame X axis
    runs along the corridor, Y across it). Only this pair has been checked;
    the X-normal end walls are not handled here."""
    return [p for p in planes if p["orientation"] == "wall" and axis_of(p) == 1]


def find_floor_z(planes):
    floor_ceiling = [p for p in planes if p["orientation"] == "floor_ceiling"]
    if not floor_ceiling:
        raise SystemExit("no floor/ceiling plane in --planes-aligned -- cannot find floor height")
    return min(p["centroid_3d"][2] for p in floor_ceiling)


def wall_occupancy_grid(wall, centers, voxel_size, origin, wall_tol_m):
    """2D (x, z) occupancy raster of every voxel within wall_tol_m of this
    wall's plane. Returns (grid, lo_u, lo_v, n_near) -- grid is None if no
    voxel is near this wall at all."""
    normal = np.asarray(wall["normal"], dtype=float)
    normal = normal / np.linalg.norm(normal)
    dist = centers @ normal + wall["d"]
    near = np.abs(dist) <= wall_tol_m
    c = centers[near]
    if len(c) == 0:
        return None, 0, 0, 0

    u_idx = np.round((c[:, 0] - origin[0]) / voxel_size).astype(np.int64)
    v_idx = np.round((c[:, 2] - origin[2]) / voxel_size).astype(np.int64)
    lo_u, lo_v = int(u_idx.min()), int(v_idx.min())
    shape = (int(u_idx.max()) - lo_u + 1, int(v_idx.max()) - lo_v + 1)
    grid = np.zeros(shape, dtype=bool)
    grid[u_idx - lo_u, v_idx - lo_v] = True
    return grid, lo_u, lo_v, len(c)


def find_wall_holes(wall, centers, voxel_size, origin, floor_z,
                     wall_tol_m=WALL_TOL_M, floor_margin_m=FLOOR_MARGIN_M,
                     closing_iterations=CLOSING_ITERATIONS,
                     min_hole_cells=MIN_HOLE_CELLS, max_hole_cells=MAX_HOLE_CELLS):
    """Candidate window holes on one wall. Returns a list of dicts:
    {wall_id, size, x_range, z_range, area_m2, cell_centers (K,3)}."""
    grid, lo_u, lo_v, n_near = wall_occupancy_grid(wall, centers, voxel_size, origin, wall_tol_m)
    wall_y = wall["centroid_3d"][1]
    print(f"wall {wall['id']}: y~{wall_y:.3f}, {n_near} near-wall voxels (tol {wall_tol_m} m)")
    if grid is None:
        return []
    print(f"  footprint bbox: {grid.shape[0]}x{grid.shape[1]} cells, "
          f"{100 * grid.mean():.1f}% occupied")

    closed = ndimage.binary_closing(grid, structure=CLOSING_STRUCTURE, iterations=closing_iterations)
    labels, n = ndimage.label(~closed, structure=BACKGROUND_STRUCTURE)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)

    holes = []
    for lbl in range(1, n + 1):
        if lbl in border:
            continue
        cells = np.argwhere(labels == lbl)
        size = len(cells)
        if not (min_hole_cells <= size <= max_hole_cells):
            continue
        cu = cells[:, 0] + lo_u
        cv = cells[:, 1] + lo_v
        xw = cu * voxel_size + origin[0]
        zw = cv * voxel_size + origin[2]
        if zw.min() <= floor_z + floor_margin_m:
            print(f"  hole: {size} cells, x[{xw.min():.2f},{xw.max():.2f}] "
                  f"z[{zw.min():.2f},{zw.max():.2f}] -- EXCLUDED (touches floor)")
            continue
        print(f"  hole: {size} cells, x[{xw.min():.2f},{xw.max():.2f}] "
              f"z[{zw.min():.2f},{zw.max():.2f}] -- KEPT (candidate window)")
        cell_centers = np.stack([xw, np.full(size, wall_y), zw], axis=1)
        holes.append({
            "wall_id": wall["id"],
            "size": size,
            "x_range": [float(xw.min()), float(xw.max())],
            "z_range": [float(zw.min()), float(zw.max())],
            "area_m2": float(size * voxel_size * voxel_size),
            "cell_centers": cell_centers,
        })
    return holes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxels", type=Path,
                     default=Path(__file__).resolve().parent.parent / "AlignedOctree" / "voxels.npz",
                     help="AlignedOctree's voxels.npz -- thresholds here are tuned for a "
                          "15cm grid, retune (--closing-iterations etc.) for a materially "
                          "different voxel size")
    ap.add_argument("--planes-aligned", type=Path,
                     default=Path(__file__).resolve().parent.parent / "AlignedOctree" / "planes_aligned.json")
    ap.add_argument("--wall-tol-m", type=float, default=WALL_TOL_M,
                     help="how close to the wall plane counts as near-wall structure")
    ap.add_argument("--floor-margin-m", type=float, default=FLOOR_MARGIN_M,
                     help="holes reaching within this height of the floor are excluded "
                          "(doorway/passage, not an elevated window)")
    ap.add_argument("--closing-iterations", type=int, default=CLOSING_ITERATIONS,
                     help="morphological closing strength before the enclosed-hole test "
                          "(bridges real coverage porosity; too high erases real windows too)")
    ap.add_argument("--min-hole-cells", type=int, default=MIN_HOLE_CELLS,
                     help="below this: dropout speck, not a real window")
    ap.add_argument("--max-hole-cells", type=int, default=MAX_HOLE_CELLS,
                     help="above this: probably not a window (open end, big structural gap)")
    ap.add_argument("--out", type=Path, default=Path("window_gaps.npz"))
    ap.add_argument("--windows-out", type=Path, default=Path("windows.json"))
    args = ap.parse_args()

    voxels = np.load(args.voxels)
    centers = voxels["centers"]
    voxel_size = float(voxels["voxel_size"])
    origin = voxels["origin"]
    print(f"loaded {args.voxels}: {len(centers)} voxels, voxel_size={voxel_size:.4f} m")

    planes = json.loads(args.planes_aligned.read_text())["planes"]
    floor_z = find_floor_z(planes)
    target_walls = find_target_walls(planes)
    print(f"target (north/south) wall(s): {[p['id'] for p in target_walls]}, floor z={floor_z:.3f}")
    if not target_walls:
        raise SystemExit("no north/south (Y-normal) wall found in --planes-aligned")

    all_holes = []
    for wall in target_walls:
        all_holes.extend(find_wall_holes(
            wall, centers, voxel_size, origin, floor_z,
            wall_tol_m=args.wall_tol_m, floor_margin_m=args.floor_margin_m,
            closing_iterations=args.closing_iterations,
            min_hole_cells=args.min_hole_cells, max_hole_cells=args.max_hole_cells))

    print(f"\ntotal windows found: {len(all_holes)}, "
          f"total synthetic glass voxels: {sum(h['size'] for h in all_holes)}")

    if all_holes:
        fill_centers = np.concatenate([h["cell_centers"] for h in all_holes], axis=0)
        window_id = np.concatenate([np.full(h["size"], i) for i, h in enumerate(all_holes)])
        wall_id = np.concatenate([np.full(h["size"], h["wall_id"]) for h in all_holes])
    else:
        fill_centers = np.empty((0, 3))
        window_id = np.empty((0,), dtype=np.int64)
        wall_id = np.empty((0,), dtype=np.int64)

    np.savez(
        args.out,
        centers=fill_centers,
        window_id=window_id,
        wall_id=wall_id,
        voxel_size=voxel_size,
        origin=origin,
    )
    print(f"wrote {args.out}")

    windows_summary = [{
        "window_id": i,
        "wall_id": h["wall_id"],
        "n_cells": h["size"],
        "area_m2": round(h["area_m2"], 3),
        "x_range": [round(v, 3) for v in h["x_range"]],
        "z_range": [round(v, 3) for v in h["z_range"]],
    } for i, h in enumerate(all_holes)]
    args.windows_out.write_text(json.dumps({
        "schema": "window_gap_fill/v1",
        "generated_by": "find_window_gaps.py",
        "source_voxels": str(args.voxels),
        "source_planes_aligned": str(args.planes_aligned),
        "params": {
            "wall_tol_m": args.wall_tol_m, "floor_margin_m": args.floor_margin_m,
            "closing_iterations": args.closing_iterations,
            "min_hole_cells": args.min_hole_cells, "max_hole_cells": args.max_hole_cells,
        },
        "windows": windows_summary,
    }, indent=2))
    print(f"wrote {args.windows_out}")


if __name__ == "__main__":
    main()
