"""OcTree — sample a TUM-FACADE point cloud into voxels and visualize it.

Loads a .las (extract one first with extract_sample.py), then either prints
octree/voxel statistics, launches the interactive GUI, or renders a headless
screenshot.

Usage:
    py main.py                          # launch the interactive viewer (default sample)
    py main.py --info                   # print point/octree/voxel stats, no GUI
    py main.py --screenshot out.png     # headless render to an image
    py main.py --smooth                 # flatten to planar surfaces (auto-aligned axis 'u')
    py main.py --smooth --smooth-axis v # the other main wall direction
    py main.py --export-openstudio surfaces.json --smooth-axis u
    py main.py --selftest               # octree<->voxel consistency checks
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from octree import (
    build_octree,
    class_name,
    leaf_voxels,
    level_counts,
    load_las,
    root_extent,
    voxelize_octree,
)

DEFAULT_LAS = Path(__file__).resolve().parent / "data" / "DEBY_LOD2_4959459.las"


def parse_args():
    p = argparse.ArgumentParser(description="Octree voxel sampling + viewer")
    p.add_argument("--las", default=str(DEFAULT_LAS), help="Path to a .las file")
    p.add_argument(
        "--voxel-size", type=float, default=0.20,
        help="Initial voxel edge in metres (GUI slider range 0.05-1.0)",
    )
    p.add_argument(
        "--min-count", type=int, default=1,
        help="Initial minimum points/voxel filter threshold (1-10)",
    )
    p.add_argument(
        "--filter", action="store_true",
        help="Start with the minimum-points filter enabled",
    )
    p.add_argument(
        "--smooth", action="store_true",
        help="Flatten voxels to a plane (OpenStudio surfaces) instead of showing cubes",
    )
    p.add_argument(
        "--smooth-axis", choices=("x", "y", "z", "u", "v"), default="u",
        help="Surface-normal axis to flatten along: literal x/y/z, or the "
        "PCA-auto-aligned dominant wall direction 'u' (default) / "
        "perpendicular direction 'v'",
    )
    p.add_argument(
        "--rotation-deg", type=float, default=None,
        help="Manual yaw override in degrees, only valid with --smooth-axis u/v "
        "(default: auto-detected via PCA on the horizontal footprint)",
    )
    p.add_argument(
        "--offset-method", choices=("mode", "median", "outer"), default="mode",
        help="How the plane offset is chosen along the axis (default mode)",
    )
    p.add_argument(
        "--tolerance", type=int, default=3,
        help="Half-width in voxels of the band snapped onto the plane (default 3)",
    )
    p.add_argument(
        "--export-openstudio", default=None, metavar="PATH.json",
        help="Smooth and write OpenStudio-friendly polygon JSON, then exit",
    )
    # RANSAC dominant-plane -> 2D wall raster (planes.py).
    p.add_argument(
        "--planes", action="store_true",
        help="RANSAC dominant-plane detection: flatten to a per-wall 2D "
        "temperature raster (heatmap) and the wall rectangle, instead of cubes",
    )
    p.add_argument(
        "--ransac-threshold", type=float, default=0.10, metavar="M",
        help="Inlier distance for RANSAC plane fitting, in metres (default 0.10)",
    )
    p.add_argument(
        "--ransac-iters", type=int, default=1000,
        help="RANSAC hypothesis iterations (default 1000)",
    )
    p.add_argument(
        "--ransac-on", choices=("centroids", "points"), default="centroids",
        help="Fit the plane on voxel centroids (fast proxy, default) or raw points",
    )
    p.add_argument(
        "--plane-rank", type=int, default=1, metavar="N",
        help="Which detected plane to use, 1-based by inlier count: 1 = biggest "
        "wall (default), 2 = the next-biggest (often the perpendicular facade)",
    )
    p.add_argument(
        "--target-normal", default=None, metavar="X,Y,Z",
        help="Pick the plane whose normal is closest to this direction "
        "(e.g. 1,0,0), overriding --plane-rank",
    )
    p.add_argument(
        "--orientation", choices=("any", "vertical", "horizontal"), default="any",
        help="Restrict plane choice: 'vertical' = facades only, 'horizontal' = "
        "floors/roofs only (default any)",
    )
    p.add_argument(
        "--keep-ground", action="store_true",
        help="Skip ground removal before RANSAC (keeps the terrain slab)",
    )
    p.add_argument(
        "--ground-band", type=float, default=0.5, metavar="M",
        help="Half-thickness of the removed horizontal ground slab (default 0.5 m)",
    )
    p.add_argument(
        "--raster-cell", type=float, default=None, metavar="M",
        help="Wall raster cell size in metres (default: the voxel size)",
    )
    p.add_argument("--seed", type=int, default=0, help="RANSAC / synthetic-field RNG seed")
    p.add_argument(
        "--temperature-dim", default=None, metavar="NAME",
        help="LAS extra-dimension name holding per-point temperature",
    )
    p.add_argument(
        "--synthetic-temp", action="store_true",
        help="Force the synthetic temperature field even if the .las has one",
    )
    p.add_argument(
        "--export-wall", default=None, metavar="PATH.json",
        help="Run the RANSAC pipeline and write the wall polygon JSON + "
        "<stem>_raster.npy + <stem>_raster.png + <stem>_qc.json, then exit",
    )
    p.add_argument("--info", action="store_true", help="Print statistics and exit (no GUI)")
    p.add_argument("--screenshot", default=None, help="Render to this image file (headless)")
    p.add_argument("--selftest", action="store_true", help="Run octree<->voxel checks and exit")
    p.add_argument(
        "--plane-selftest", action="store_true",
        help="Run RANSAC/basis/raster/rectangle checks on a synthetic wall and exit",
    )
    p.add_argument("--max-points", type=int, default=2_000_000, help="Max points drawn in the GUI")
    return p.parse_args()


def print_info(pc):
    lo, hi = pc.bounds
    extent = root_extent(pc.points)
    print(f"\nPoints: {len(pc):,}")
    print(f"Bounds min: {np.round(lo, 2)}")
    print(f"Bounds max: {np.round(hi, 2)}")
    print(f"Cubic root extent: {extent:.2f} m")

    ids, counts = np.unique(pc.labels, return_counts=True)
    print("\nSemantic classes present:")
    for cid, cnt in sorted(zip(ids.tolist(), counts.tolist()), key=lambda x: -x[1]):
        print(f"  {cid:>2} {class_name(cid):<22} {cnt:>10,}  ({cnt / len(pc):5.1%})")

    print("\nOctree occupied nodes per depth (root=depth 0):")
    root = build_octree(pc.points, max_depth=8)
    for d, n in enumerate(level_counts(root)):
        print(f"  depth {d}:  {n:>8,} nodes   (<= 8^{d} = {8**d:,})")


def selftest(pc):
    """Occupied octree leaves at depth d == voxelizer voxels at the matching size."""
    print("\nOctree <-> voxelizer consistency:")
    ok = True
    for d in range(3, 8):
        n_vox = len(voxelize_octree(pc.points, pc.labels, d))
        root = build_octree(pc.points, max_depth=d)
        centers, _ = leaf_voxels(root)
        n_leaf = len(centers)
        match = n_vox == n_leaf
        ok &= match
        print(f"  depth {d}: voxelizer={n_vox:>7,}  octree_leaves={n_leaf:>7,}  {'OK' if match else 'MISMATCH'}")

    # Monotonicity: finer voxels -> more voxels, but never more than points.
    ns = [len(voxelize_octree(pc.points, pc.labels, d)) for d in range(3, 9)]
    mono = all(a <= b for a, b in zip(ns, ns[1:])) and ns[-1] <= len(pc)
    ok &= mono
    print(f"  voxel counts by depth 3..8: {ns}  monotonic&bounded: {'OK' if mono else 'FAIL'}")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def plane_selftest():
    """RANSAC/basis/raster/rectangle checks on a synthetic vertical wall."""
    from octree import (
        fit_plane_ransac,
        min_area_rect,
        plane_basis,
        project_to_plane,
        rasterize,
    )

    print("\nRANSAC dominant-plane self-test (synthetic vertical wall):")
    rng = np.random.default_rng(0)
    n_pts = 5000
    y = rng.uniform(0.0, 8.0, n_pts)   # wall width  (u ~ 8 m)
    z = rng.uniform(0.0, 12.0, n_pts)  # wall height (v ~ 12 m)
    x = 5.0 + rng.normal(0.0, 0.01, n_pts)  # tight about the plane x = 5
    pts = np.column_stack([x, y, z])
    injected = 20.0
    vals = np.full(n_pts, injected)

    plane = fit_plane_ransac(pts, threshold=0.05, iters=200, seed=1)
    normal_ok = abs(abs(plane.normal @ np.array([1.0, 0.0, 0.0])) - 1.0) < 1e-2

    inl = pts[plane.inliers]
    origin, e_u, e_v, n = plane_basis(plane.normal, inl)
    u, v, d = project_to_plane(inl, origin, e_u, e_v, n)
    offset_ok = float(np.abs(d).max()) < 0.05

    r = rasterize(u, v, vals[plane.inliers], 0.5, offsets=d)
    finite = r.values[np.isfinite(r.values)]
    mean_ok = bool(np.allclose(finite, injected, atol=1e-6))

    rect = min_area_rect(np.column_stack([u, v]))
    w = float(np.linalg.norm(rect[1] - rect[0]))
    h = float(np.linalg.norm(rect[3] - rect[0]))
    area = w * h
    area_ok = abs(area - 96.0) < 0.15 * 96.0  # 8 x 12 = 96 m^2

    for label, ok in [
        ("normal recovered (|n.x|~1)", normal_ok),
        ("inlier offsets d ~ 0", offset_ok),
        (f"raster mean == {injected:.0f}", mean_ok),
        (f"rect area ~ 96 m^2 (got {area:.1f})", area_ok),
    ]:
        print(f"  {'OK  ' if ok else 'FAIL'} {label}")
    ok = normal_ok and offset_ok and mean_ok and area_ok
    print("PLANE SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def _parse_normal(text):
    """Parse a 'X,Y,Z' target-normal string into a 3-tuple, or None."""
    if not text:
        return None
    parts = [float(x) for x in text.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError("--target-normal must be 'X,Y,Z' (three numbers)")
    return tuple(parts)


def _resolve_temperature(pc, force_synthetic, seed):
    """Per-point temperature to average into the raster (deg C), + its source.

    The .las value is used when present unless --synthetic-temp forces the
    synthetic field. Returns (array_or_None, source_label): None lets the
    downstream planes path fall back to a synthetic field on the centroids.
    """
    from octree import synthetic_temperature

    if force_synthetic:
        return synthetic_temperature(pc.points, seed=seed), "synthetic (forced)"
    if pc.temperature is not None:
        return pc.temperature, "las extra-dim"
    return synthetic_temperature(pc.points, seed=seed), "synthetic (no thermal in cloud)"


def export_wall(pc, args):
    """Run the RANSAC pipeline and write JSON + raster .npy/.png + QC json."""
    import json

    from octree import (
        filter_by_count,
        run_dominant_plane,
        save_raster_npy,
        synthetic_temperature,
        to_openstudio_json,
        voxelize,
        wall_qc_dict,
        wall_to_surface,
    )
    from octree.viewer import render_wall_screenshot

    temp, tsource = _resolve_temperature(pc, args.synthetic_temp, args.seed)
    cell = args.raster_cell if args.raster_cell else args.voxel_size
    effective_min = args.min_count if args.filter else 1
    common = dict(
        threshold=args.ransac_threshold, iters=args.ransac_iters, seed=args.seed,
        keep_ground=args.keep_ground, ground_band=args.ground_band, raster_cell=cell,
        rank=args.plane_rank, target_normal=_parse_normal(args.target_normal),
        orientation=args.orientation,
    )

    if args.ransac_on == "points":
        wall = run_dominant_plane(pc.points, temp, labels=pc.labels, **common)
    else:
        grid = voxelize(pc.points, pc.labels, args.voxel_size, values=temp)
        grid = filter_by_count(grid, effective_min)
        vals = grid.values if grid.values is not None else synthetic_temperature(
            grid.centers, seed=args.seed
        )
        wall = run_dominant_plane(grid.centers, vals, labels=grid.labels, **common)

    base = Path(args.export_wall)
    stem = base.with_suffix("")
    npy_path = stem.with_name(stem.name + "_raster.npy")
    png_path = stem.with_name(stem.name + "_raster.png")
    qc_path = stem.with_name(stem.name + "_qc.json")

    to_openstudio_json(wall_to_surface(wall), base)
    save_raster_npy(wall, npy_path)
    qc_path.write_text(json.dumps(wall_qc_dict(wall), indent=2), encoding="utf-8")
    render_wall_screenshot(wall, str(png_path))

    w, h = wall.rect_dims
    n = wall.normal
    print(
        f"Temperature: {tsource}. RANSAC on {args.ransac_on} "
        f"(threshold {args.ransac_threshold:.2f} m).\n"
        f"Selected plane {wall.rank}/{wall.n_candidates} "
        f"(by {'target-normal' if args.target_normal else 'rank'}, "
        f"orientation={args.orientation}).\n"
        f"Plane normal ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f}); "
        f"{wall.n_inliers:,}/{wall.n_fitted:,} inliers, ground -{wall.n_ground:,}.\n"
        f"Wall rectangle {w:.2f} x {h:.2f} m; raster {wall.raster.shape[0]}x"
        f"{wall.raster.shape[1]} @ {cell:.2f} m "
        f"({wall.raster.occupancy:.0%} occupied).\n"
        f"QC protrusions: {wall.offset_stats['n_protrusions']:,} "
        f"({wall.offset_stats['protrusion_frac']:.1%}) beyond "
        f"{wall.offset_stats['protrusion_band_m']:.2f} m.\n"
        f"Wrote {base}, {npy_path.name}, {png_path.name}, {qc_path.name}"
    )
    return 0


def main():
    args = parse_args()

    if args.plane_selftest:
        return 0 if plane_selftest() else 1

    print(f"Loading {args.las} ...")
    pc = load_las(args.las, temperature_dim=args.temperature_dim)

    if args.info:
        print_info(pc)
        return 0

    if args.selftest:
        return 0 if selftest(pc) else 1

    if args.export_wall:
        return export_wall(pc, args)

    effective_min = args.min_count if args.filter else 1

    # Per-point temperature for the planes raster (None -> synthetic fallback
    # on the centroids downstream); forced synthetic overrides a .las field.
    temperature = None
    if args.planes:
        temperature, tsource = _resolve_temperature(pc, args.synthetic_temp, args.seed)
        print(f"Temperature source: {tsource}")

    if args.export_openstudio:
        from octree import smooth_surface, to_openstudio_json, voxelize
        from octree.voxelizer import filter_by_count

        grid = voxelize(pc.points, pc.labels, args.voxel_size)
        grid = filter_by_count(grid, effective_min)
        surface = smooth_surface(
            grid, args.smooth_axis, args.offset_method, args.tolerance,
            rotation_deg=args.rotation_deg,
        )
        path = to_openstudio_json(surface, args.export_openstudio)
        yaw_note = f" (yaw {surface.rotation_deg:.1f} deg)" if args.smooth_axis in ("u", "v") else ""
        print(
            f"Smoothed along {args.smooth_axis}{yaw_note} (plane {surface.plane_coord:+.2f} m): "
            f"{surface.n_inliers:,} inliers, {surface.n_deviations:,} deviations, "
            f"{len(surface.subsurfaces)} sub-surfaces, {surface.n_polygons} polygons.\n"
            f"Wrote {path}"
        )
        return 0

    if args.screenshot:
        from octree.viewer import render_screenshot

        result = render_screenshot(
            pc.points, pc.labels, args.screenshot,
            voxel_size=args.voxel_size, max_points=args.max_points,
            min_count=effective_min, smooth=args.smooth, smooth_axis=args.smooth_axis,
            offset_method=args.offset_method, tolerance=args.tolerance,
            rotation_deg=args.rotation_deg,
            planes=args.planes, temperature=temperature,
            ransac_threshold=args.ransac_threshold, ransac_iters=args.ransac_iters,
            raster_cell=args.raster_cell, keep_ground=args.keep_ground,
            ground_band=args.ground_band, seed=args.seed,
            plane_rank=args.plane_rank, target_normal=_parse_normal(args.target_normal),
            orientation=args.orientation,
        )
        print(f"Wrote {args.screenshot}  (voxel {args.voxel_size:.2f} m)")
        return 0

    from octree.viewer import launch

    print("Opening viewer — sliders: 'voxel size (m)', 'min points/voxel'; "
          "checkboxes: points / filter / smooth / planes (+ 'next plane' to "
          "cycle facades); axis: u / v / z.")
    launch(
        pc.points, pc.labels, voxel_size=args.voxel_size, max_points=args.max_points,
        min_count=args.min_count, filter_on=args.filter,
        smooth_on=args.smooth, smooth_axis=args.smooth_axis,
        offset_method=args.offset_method, tolerance=args.tolerance,
        rotation_deg=args.rotation_deg,
        planes_on=args.planes, temperature=temperature,
        ransac_threshold=args.ransac_threshold, ransac_iters=args.ransac_iters,
        raster_cell=args.raster_cell, keep_ground=args.keep_ground,
        ground_band=args.ground_band, seed=args.seed,
        plane_rank=args.plane_rank, target_normal=_parse_normal(args.target_normal),
        orientation=args.orientation,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
